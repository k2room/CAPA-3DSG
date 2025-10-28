"""
    Open-vocabulary 2D object/part detection in each 2D image.
    - use CAPA.yaml for configuration.
    - example usage: 
        $ python scripts/2D_detection.py --scene_id 0kitchen/video0 --dataset FunGraph3D
"""
import os, argparse, sys
from pathlib import Path
from typing import Any, List
from PIL import Image
import cv2
import json
import imageio
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pickle
import gzip
import open_clip
import torch
import torchvision
import supervision as sv
from tqdm import trange
import torchvision.transforms as TS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from dataloader.datasets_common import get_dataset
from utils.vis import vis_result_fast
from utils.utils import load_yaml
from utils.knowledge import load_knowledge, curate_tags
from utils.vlp import VLPart, to_sv_detections

try: 
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: SAM")
    raise e

try:
    from ram.models import ram
    from ram import inference_ram
except ImportError as e:
    print("Import Error: RAM")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)

def get_parser() -> argparse.ArgumentParser:
    # Load default config
    print(f"Loading default config file: {Path(__file__).parent.parent / 'configs' / 'CAPA.yaml'}")
    cfg = load_yaml(Path(__file__).parent.parent / 'configs' / 'CAPA.yaml')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["FunGraph3D", "SceneFun3D/dev", "SceneFun3D/test", "PADO"])
    
    parser.add_argument("--start", type=int, default=cfg["start"])
    parser.add_argument("--end", type=int, default=cfg["end"])
    parser.add_argument("--stride", type=int, default=cfg["stride"])
    parser.add_argument("--save_folder_name", type=str, default=cfg["save_folder_name"])

    parser.add_argument("--desired-height", type=int, default=cfg["desired_height"])
    parser.add_argument("--desired-width", type=int, default=cfg["desired_width"])

    parser.add_argument("--tagger", type=str, default=cfg["tagger"], choices=["ram", "none"], help="If none, no tagging and detection will be used and the SAM will be run in dense sampling mode. ")
    parser.add_argument("--detector", type=str, default=cfg["detector"], choices=["gdino", "yolo", "vlp"])
    parser.add_argument("--add_bg_classes", type=bool, default=cfg["add_bg_classes"], help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    parser.add_argument("--accumu_classes", type=bool, default=cfg["accumu_classes"], help="if set, the class set will be accumulated over frames")
    
    parser.add_argument("--device", type=str, default=cfg["device"])
    parser.add_argument("--exp_suffix", type=str, default=cfg["exp_suffix"], help="The suffix of the folder that the results will be saved to. ")

    parser.add_argument("--gdino_config_path", type=str, default=cfg["gdino_config_path"])
    parser.add_argument("--gdino_ckpt", type=str, default=cfg["gdino_ckpt"])
    parser.add_argument("--vlpart_config_path", type=str, default=cfg["vlpart_config_path"])
    parser.add_argument("--vlpart_ckpt", type=str, default=cfg["vlpart_ckpt"])
    parser.add_argument("--sam_enc_version", type=str, default=cfg["sam_enc_version"])
    parser.add_argument("--sam_ckpt", type=str, default=cfg["sam_ckpt"])
    parser.add_argument("--ram_ckpt", type=str, default=cfg["ram_ckpt"])
    parser.add_argument("--knowledge_path", type=str, default=cfg["knowledge_path"], help="Path to the object/part knowledge JSON file. ")

    parser.add_argument("--skip_bg", type=bool, default=cfg["skip_bg"], help="If set, skip the background classes during detection and tagging. ")
    parser.add_argument("--bg_classes", type=list, default=cfg["bg_classes"], help="List of background classes to skip if --skip_bg is set. ")

    args = parser.parse_args()
    if args.dataset == "FunGraph3D":
        args.dataset_root = Path(cfg["FUNGRAPH3D_root"])
        args.dataset_config = cfg["FUNGRAPH3D_config_path"]
    elif args.dataset == "SceneFun3D/dev":
        args.dataset_root = Path(cfg["SCENEFUN3D_root"] / "dev")
        args.dataset_config = cfg["SCENEFUN3D_config"]
    elif args.dataset == "SceneFun3D/test":
        args.dataset_root = Path(cfg["SCENEFUN3D_root"] / "test")
        args.dataset_config = cfg["SCENEFUN3D_config"]
    elif args.dataset == "PADO":
        args.dataset_root = Path(cfg["PADO_root"])
        args.dataset_config = cfg["PADO_config_path"]
    else:
        raise ValueError("Unknown dataset: ", args.dataset)

    if args.detector == "gdino":
        gdino_cfg = cfg.get("gdino", {})
        args.box_threshold = gdino_cfg["box_threshold"]
        args.text_threshold = gdino_cfg["text_threshold"]
        args.nms_threshold = gdino_cfg["nms_threshold"]
        args.part_box_threshold = gdino_cfg["part_box_threshold"]
        args.part_text_threshold = gdino_cfg["part_text_threshold"]
        args.part_nms_threshold = gdino_cfg["part_nms_threshold"]
    elif args.detector == "yolo":
        yolo_cfg = cfg.get("yolo", {})
        args.box_threshold = gdino_cfg["box_threshold"]
        args.text_threshold = gdino_cfg["text_threshold"]
        args.nms_threshold = gdino_cfg["nms_threshold"]
    elif args.detector == "vlp":
        vlp_cfg = cfg.get("vlp", {})
        args.score_threshold = vlp_cfg["score_threshold"]
        args.nms_threshold = vlp_cfg["nms_threshold"]
        args.part_nms_threshold = vlp_cfg["part_nms_threshold"]
    else:
        raise ValueError("Unknown detector: ", args.detector)
    
    return args

def get_sam_predictor(args: argparse.Namespace) -> SamPredictor:
    sam = sam_model_registry[args.sam_enc_version](checkpoint=args.sam_ckpt)
    sam.to(args.device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def get_sam_mask_generator(args: argparse.Namespace) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[args.sam_enc_version](checkpoint=args.sam_ckpt)
    sam.to(args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=12,
        points_per_batch=144,
        pred_iou_thresh=0.88,
        stability_score_threshold=0.95,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    return mask_generator

def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def get_sam_segmentation_dense(model: Any, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
        The SAM based on automatic mask generation, without bbox prompting
        
        Args:
            model: The mask generator or the YOLO model
            image: (H, W, 3), in RGB color space, in range [0, 255]
            
        Returns:
            mask: (N, H, W)
            xyxy: (N, 4)
            conf: (N,)
    '''
    results = model.generate(image)
    mask = []
    xyxy = []
    conf = []
    for r in results:
        mask.append(r["segmentation"])
        r_xyxy = r["bbox"].copy()
        # Convert from xyhw format to xyxy format
        r_xyxy[2] += r_xyxy[0]
        r_xyxy[3] += r_xyxy[1]
        xyxy.append(r_xyxy)
        conf.append(r["predicted_iou"])
    mask = np.array(mask)
    xyxy = np.array(xyxy)
    conf = np.array(conf)
    return mask, xyxy, conf

    
def main(args: argparse.Namespace):

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=args.dataset_config,
        start=args.start,
        end=args.end,
        stride=args.stride,
        basedir=args.dataset_root,
        sequence=args.scene_id,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
        device="cpu",
        dtype=torch.float,
    )

    # Initialize the detection models
    if args.detector == "yolo": # Todo: implement YOLO + SAM
        try:
            from ultralytics import YOLO, SAM as ULTRA_SAM
        except ImportError as e:
            print("Import Error: YOLOv8 / YOLOv8-SAM from ultralytics")
            raise e

        print("LOAD CLIP MODELS...", end="")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to(args.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("DONE.")

        print("LOAD YOLO MODELS...", end="")
        yolo_model = YOLO('yolov8l-world.pt')
        print("DONE.")

    elif args.detector == "gdino":
        try: 
            from groundingdino.util.inference import Model as GDINO
        except ImportError as e:
            print("Import Error: Grounding DINO")
            raise e

        print("LOAD CLIP MODELS...", end="")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to(args.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("DONE.")

        print("LOAD GROUNDING DINO MODEL...", end="")
        gdino_model = GDINO(
            model_config_path=args.gdino_config_path, 
            model_checkpoint_path=args.gdino_ckpt, 
            device=args.device
        )
        print("DONE.")

    elif args.detector == "vlp":
        print("LOAD VLPART MODELS...", end="")
        vlpart = VLPart(
            model_path=args.vlpart_ckpt, 
            config_file=args.vlpart_config_path, 
            device=args.device, 
            score_thresh=args.score_threshold)
        print("DONE.")

    else:
        raise ValueError("Unknown detector: ", args.detector)

    # Initialize the SAM model
    if args.tagger == "none":
        mask_generator = get_sam_mask_generator(args)
    else:
        sam_predictor = get_sam_predictor(args)
    
    # Initialize the tagging model
    if args.tagger == "ram":
        classes = None
        print("LOAD RAM MODEL...", end="")
        tagging_model = ram(pretrained=args.ram_ckpt, image_size=384, vit='swin_l')
        tagging_model = tagging_model.eval().to(args.device)
        print("DONE.")
        
        # initialize Tag2Text
        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            ])
    elif args.tagger == "none":
        print("No tagging model will be used. Use SAM in dense sampling mode.")
        classes = ['item']
    else:
        raise ValueError("Unknown args.tagger: ", args.tagger)
        
    save_name = f"{args.tagger}"
    if args.exp_suffix:
        save_name += f"_{args.exp_suffix}"

    global_classes = set()
    obj_classes = set()
    part_classes = set()
    knowledge = load_knowledge(args)

    #################################### Main loop ####################################
    for idx in trange(len(dataset)):
    
        color_path = dataset.color_paths[idx]
        color_path = Path(color_path)           # .../<scene>/rgb/frame_00000.jpg

        vis_save_path = color_path.parent.parent / args.save_folder_name / f"gsa_vis_{save_name}" / color_path.name
        detections_save_path = color_path.parent.parent / args.save_folder_name / f"gsa_detections_{save_name}" / color_path.name
        detections_save_path = detections_save_path.with_suffix(".pkl.gz")
        
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(detections_save_path), exist_ok=True)
        
        # opencv can't read Path objects, so convert to str
        color_path = str(color_path)
        vis_save_path = str(vis_save_path)
        detections_save_path = str(detections_save_path)
        
        image = cv2.imread(color_path) # This will in BGR color space
        if hasattr(dataset, 'camera_axis'):
            if dataset.camera_axis == 'Left':   # rotate it
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
        image_pil = Image.fromarray(image_rgb)
        
        ########### Get the tags using RAM ###########
        if args.detector == "vlp" or args.detector == "gdino":
            if args.tagger == "ram":
                raw_image = image_pil.resize((384, 384))
                raw_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)
                ram_out = inference_ram(raw_image, tagging_model)[0]

                raw_tags = [t.strip() for t in ram_out.split(" | ")]
                obj_tags, part_tags = curate_tags(raw_tags, knowledge, args)

                # add classes list to global classes
                obj_classes.update(obj_tags)
                part_classes.update(part_tags)
                global_classes.update(obj_tags+part_tags)
        
            if args.accumu_classes: # Use all the classes that have been seen so far
                classes = list(global_classes)
            else: # Use only the current frame's classes
                classes = obj_tags + part_tags
        else:
            print("YOLO do not use RAM tagging module.")


        ########### Detection & Segmentation ###########
        if args.tagger == "none":
            # Directly use SAM in dense sampling mode to get segmentation
            mask, xyxy, conf = get_sam_segmentation_dense(mask_generator, image_rgb)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)

            # visualize results 
            annotated_image, labels = vis_result_fast(image, detections, classes, instance_random_color=True)
            cv2.imwrite(vis_save_path, annotated_image)

        elif args.detector == "yolo":
            # TODO: Implement YOLO + SAM
            print("YOLO + SAM not implemented yet.")
            raise NotImplementedError

        elif args.detector == "gdino":
            # TODO: Implement GroundingDINO + SAM
            print("GroundingDINO + SAM not implemented yet.")
            raise NotImplementedError

            # # Using GroundingDINO to detect and SAM to segment
            # detections = grounding_dino_model.predict_with_classes(
            #     image=image,
            #     classes=classes,
            #     box_threshold=args.box_threshold,
            #     text_threshold=args.text_threshold,
            # )
        
            # if len(detections.class_id) > 0:
            #     # Non-Maximum Suppression
            #     nms_idx = torchvision.ops.nms(
            #         torch.from_numpy(detections.xyxy), 
            #         torch.from_numpy(detections.confidence), 
            #         args.nms_threshold
            #     ).numpy().tolist()
            #     # print(f"After NMS: {len(detections.xyxy)} boxes")

            #     detections.xyxy = detections.xyxy[nms_idx]
            #     detections.confidence = detections.confidence[nms_idx]
            #     detections.class_id = detections.class_id[nms_idx]
                
            #     # Somehow some detections will have class_id=-1, remove them
            #     valid_idx = detections.class_id != -1
            #     detections.xyxy = detections.xyxy[valid_idx]
            #     detections.confidence = detections.confidence[valid_idx]
            #     detections.class_id = detections.class_id[valid_idx]

            #            # Compute and save the clip features of detections  
            # image_crops, image_feats, text_feats = compute_clip_features(
            #     image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)

        elif args.detector == "vlp":
            ########### VLPart Detection ###########
            detections = vlpart.predict_with_classes(image=image_rgb, classes=classes)
        
            ########### Non-Maximum Suppression ###########
            if len(detections.class_id) > 0:
                valid = detections.class_id != -1
                if np.any(~valid):
                    sel0 = np.nonzero(valid)[0]
                    detections.xyxy        = detections.xyxy[sel0]
                    detections.confidence  = detections.confidence[sel0]
                    detections.class_id    = detections.class_id[sel0]
                    detections.mask = np.take(detections.mask, sel0, axis=0)
                    detections.image_crops = [detections.image_crops[i] for i in sel0]
                    detections.image_feats = np.take(detections.image_feats, sel0, axis=0)
                
                print(f"Detections before NMS: {len(detections.class_id)} boxes")
                cls_names = np.array([classes[cid] for cid in detections.class_id])  # (N,)
                is_part = np.isin(cls_names, part_classes)   # part_classes: set[str]
                is_obj  = np.isin(cls_names, obj_classes)    # obj_classes : set[str]
                is_unknown = ~(is_part | is_obj)
                tensor_boxes = torch.as_tensor(detections.xyxy, dtype=torch.float32)
                tensor_scores = torch.as_tensor(detections.confidence, dtype=torch.float32)

                keep_global = []

                if np.any(is_obj):
                    sub = np.flatnonzero(is_obj)
                    idx = torchvision.ops.nms(tensor_boxes[sub], tensor_scores[sub], float(args.nms_threshold))
                    keep_global.append(sub[idx.cpu().numpy()])

                if np.any(is_part):
                    sub = np.flatnonzero(is_part)
                    idx = torchvision.ops.nms(tensor_boxes[sub], tensor_scores[sub], float(args.part_nms_threshold))
                    keep_global.append(sub[idx.cpu().numpy()])

                if np.any(is_unknown): # unknown classes are treated with object thresholds
                    sub = np.flatnonzero(is_unknown)
                    idx = torchvision.ops.nms(tensor_boxes[sub], tensor_scores[sub], float(args.nms_threshold))
                    keep_global.append(sub[idx.cpu().numpy()])

                if len(keep_global) == 0:
                    print("No valid detections after NMS.")
                    keep_idx = np.empty((0,), dtype=int)
                else:
                    keep_idx = np.concatenate(keep_global, axis=0)

                detections.xyxy = detections.xyxy[keep_idx]
                detections.confidence = detections.confidence[keep_idx]
                detections.class_id = detections.class_id[keep_idx]
                detections.mask = np.take(detections.mask, keep_idx, axis=0)
                detections.image_crops = [detections.image_crops[i] for i in keep_idx]
                detections.image_feats = np.take(detections.image_feats, keep_idx, axis=0)

                is_obj = np.isin(
                    np.array([classes[int(cid)] for cid in detections.class_id]),
                    list(obj_classes),
                ).astype(bool)

                is_part = np.isin(
                    np.array([classes[int(cid)] for cid in detections.class_id]),
                    list(part_classes),
                ).astype(bool)

                print(f"After NMS: {len(detections.xyxy)} boxes")

            ########### Get SAM Mask ###########
            if len(detections.class_id) > 0:
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )
                image_crops = list(detections.image_crops)
                image_feats = list(detections.image_feats)
                text_feats = list(detections.text_feats[detections.class_id])
            else:
                image_crops, image_feats, text_feats, is_part, is_obj = [], [], [], [], []

        else:
            print("Cannot happen.")
            raise ValueError("Unknown args.detector: ", args.detector)


        # visualize results 
        annotated_image, labels = vis_result_fast(image, to_sv_detections(detections), classes)
        
        # save the annotated grounded-sam image
        cv2.imwrite(vis_save_path, annotated_image)
        
        # Convert the detections to a dict. The elements are in np.array
        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "is_part": is_part
        }
        print(f"Total {len(classes)} = object {np.sum(is_obj)} + part {np.sum(is_part)}")
        
        if args.tagger == "ram":
            results["tagging_caption"] = "NA"
            results["tagging_text_prompt"] = raw_tags
        
        # save the detections using pickle
        # Here we use gzip to compress the file, which could reduce the file size by 500x
        with gzip.open(detections_save_path, "wb") as f:
            pickle.dump(results, f)
    
    # save the global classes
    with open(args.dataset_root / args.scene_id / args.save_folder_name / f"gsa_classes_{save_name}.json", "w") as f:
        json.dump(list(global_classes), f)
        

if __name__ == "__main__":
    args = get_parser()
    main(args)
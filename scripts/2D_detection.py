"""
Open-vocabulary 2D object/part detection in each 2D image.
- use CAPA.yaml for configuration (Hydra).
- example:
    $ python scripts/2D_detection.py scene_id=0kitchen/video0 dataset=FunGraph3D hydra.job_logging.handlers.file.level=DEBUG
"""
# ===== imports =====
import os, sys, json, gzip, pickle, warnings, logging
from pathlib import Path
from typing import Any, List
from PIL import Image

import cv2
import imageio
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import torch
import torchvision
import supervision as sv
from tqdm import trange
import torchvision.transforms as TS

# ===== hydra / omegaconf =====
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

# register custom resolver used in logging filename: ${replace:x,/,-}
OmegaConf.register_new_resolver("replace", lambda s, a, b: str(s).replace(a, b))

warnings.simplefilter(action='ignore', category=FutureWarning)

# ===== project imports =====
from dataloader.datasets_common import get_dataset
from utils.vis import vis_result_fast
# from utils.utils import load_yaml
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

LOGGER = logging.getLogger(__name__)  # [HYDRA] use hydra-managed logger


# =========================
# Utilities (SAM helpers)
# =========================
def get_sam_predictor(cfg: DictConfig) -> SamPredictor:  # [HYDRA]
    sam = sam_model_registry[cfg.sam_enc_version](checkpoint=cfg.sam_ckpt)
    sam.to(cfg.device)
    return SamPredictor(sam)

def get_sam_mask_generator(cfg: DictConfig) -> SamAutomaticMaskGenerator:  # [HYDRA]
    sam = sam_model_registry[cfg.sam_enc_version](checkpoint=cfg.sam_ckpt)
    sam.to(cfg.device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=12,
        points_per_batch=144,
        pred_iou_thresh=0.88,
        stability_score_threshold=0.95,
        crop_n_layers=0,
        min_mask_region_area=100,
    )

def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def get_sam_segmentation_dense(model: Any, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SAM automatic mask generation (no bbox prompt)
    Returns:
        mask: (N, H, W), xyxy: (N, 4), conf: (N,)
    """
    results = model.generate(image)
    mask, xyxy, conf = [], [], []
    for r in results:
        mask.append(r["segmentation"])
        r_xyxy = r["bbox"].copy()     # xyhw -> xyxy
        r_xyxy[2] += r_xyxy[0]
        r_xyxy[3] += r_xyxy[1]
        xyxy.append(r_xyxy)
        conf.append(r["predicted_iou"])
    return np.array(mask), np.array(xyxy), np.array(conf)


# =========================
# Config enrichment (fill derived fields so downstream can use cfg like args)
# =========================
ALLOWED_DATASETS = {"FunGraph3D", "SceneFun3Ddev", "SceneFun3Dtest", "PADO"}  # [HYDRA]

def _resolve_path(p) -> str:
    """Resolve path against original CWD (Hydra changes run dir)."""
    pp = Path(str(p))
    if not pp.is_absolute():
        pp = Path(get_original_cwd()) / pp
    return str(pp)

def _enrich_cfg_inplace(cfg: DictConfig) -> None:  # [HYDRA]
    """
    - require scene_id & dataset
    - attach dataset_root / dataset_config
    - absolutize checkpoints
    - flatten detector thresholds to top-level keys (box_threshold, etc.)
    """
    if not cfg.get("scene_id") or not cfg.get("dataset"):
        raise ValueError("Both `scene_id` and `dataset` are required. e.g., scene_id=0kitchen/video0 dataset=FunGraph3D")
    if str(cfg.dataset) not in ALLOWED_DATASETS:
        raise ValueError(f"`dataset` must be one of {sorted(ALLOWED_DATASETS)}; got {cfg.dataset}")

    prev_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)

    ds = str(cfg.dataset)
    if ds == "FunGraph3D":
        cfg.dataset_root   = _resolve_path(cfg.FUNGRAPH3D_root)
        cfg.dataset_config = _resolve_path(cfg.FUNGRAPH3D_config_path)
    elif ds == "SceneFun3Ddev":
        cfg.dataset_root   = _resolve_path(Path(cfg.SCENEFUN3D_root) / "dev")
        cfg.dataset_config = _resolve_path(cfg.SCENEFUN3D_config)
    elif ds == "SceneFun3Dtest":
        cfg.dataset_root   = _resolve_path(Path(cfg.SCENEFUN3D_root) / "test")
        cfg.dataset_config = _resolve_path(cfg.SCENEFUN3D_config)
    elif ds == "PADO":
        cfg.dataset_root   = _resolve_path(cfg.PADO_root)
        pado_cfg_key = "PADO_config" if "PADO_config" in cfg else "PADO_config_path"
        cfg.dataset_config = _resolve_path(cfg[pado_cfg_key])

    # absolutize checkpoints & knowledge
    cfg.gdino_config_path  = _resolve_path(cfg.gdino_config_path)
    cfg.gdino_ckpt         = _resolve_path(cfg.gdino_ckpt)
    cfg.vlpart_config_path = _resolve_path(cfg.vlpart_config_path)
    cfg.vlpart_ckpt        = _resolve_path(cfg.vlpart_ckpt)
    cfg.sam_ckpt           = _resolve_path(cfg.sam_ckpt)
    cfg.ram_ckpt           = _resolve_path(cfg.ram_ckpt)
    cfg.knowledge_path     = _resolve_path(cfg.knowledge_path)

    # detector thresholds -> flatten
    det = str(cfg.detector)
    if det == "gdino":
        gd = cfg.gdino
        cfg.box_threshold       = gd.box_threshold
        cfg.text_threshold      = gd.text_threshold
        cfg.nms_threshold       = gd.nms_threshold
        cfg.part_box_threshold  = gd.part_box_threshold
        cfg.part_text_threshold = gd.part_text_threshold
        cfg.part_nms_threshold  = gd.part_nms_threshold
    elif det == "vlp":
        vl = cfg.vlp
        cfg.score_threshold    = vl.score_threshold
        cfg.nms_threshold      = vl.nms_threshold
        cfg.part_nms_threshold = vl.part_nms_threshold
    elif det == "yolo":
        yo = cfg.yolo
        cfg.box_threshold  = yo.box_threshold
        cfg.text_threshold = yo.text_threshold
        cfg.nms_threshold  = yo.nms_threshold
    else:
        raise ValueError(f"Unknown detector: {det}")

    OmegaConf.set_struct(cfg, prev_struct)

# =========================
# Main pipeline
# =========================
def main(cfg: DictConfig):  # [HYDRA] cfg directly
    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.desired_height,
        desired_width=cfg.desired_width,
        device="cpu",
        dtype=torch.float,
    )
    LOGGER.info(f"Dataset loaded: scene={cfg.scene_id}")
    LOGGER.info(f"Dataset loaded: {len(dataset)} frames")

    # Initialize the detection models
    if cfg.detector == "yolo":  # TODO
        try:
            from ultralytics import YOLO, SAM as ULTRA_SAM
        except ImportError as e:
            LOGGER.error("Import Error: YOLOv8 / YOLOv8-SAM from ultralytics")
            raise e

        import open_clip
        LOGGER.info("LOAD CLIP MODEL...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        LOGGER.info("DONE.")    

        LOGGER.info("LOAD YOLO MODELS...")
        yolo_model = YOLO('yolov8l-world.pt')
        LOGGER.info("DONE.")

    elif cfg.detector == "gdino": # TODO
        try:
            from groundingdino.util.inference import Model as GDINO
        except ImportError as e:
            LOGGER.error("Import Error: Grounding DINO")
            raise e

        import open_clip
        LOGGER.info("LOAD CLIP MODEL...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        LOGGER.info("DONE.")

        LOGGER.info("LOAD GROUNDING DINO MODEL...")
        gdino_model = GDINO(
            model_config_path=cfg.gdino_config_path,
            model_checkpoint_path=cfg.gdino_ckpt,
            device=cfg.device
        )
        LOGGER.info("DONE.")

    elif cfg.detector == "vlp":
        LOGGER.info("LOAD VLPART MODELS...")
        vlpart = VLPart(
            model_path=cfg.vlpart_ckpt,
            config_file=cfg.vlpart_config_path,
            device=cfg.device,
            score_thresh=cfg.score_threshold
        )
        LOGGER.info("DONE.")
    else:
        raise ValueError(f"Unknown detector: {cfg.detector}")

    # Initialize the SAM model
    if cfg.tagger == "none":
        mask_generator = get_sam_mask_generator(cfg)
    else:
        sam_predictor = get_sam_predictor(cfg)

    # Initialize the tagging model
    if cfg.tagger == "ram":
        classes = None
        LOGGER.info("LOAD RAM MODEL...")
        tagging_model = ram(pretrained=cfg.ram_ckpt, image_size=384, vit='swin_l')
        tagging_model = tagging_model.eval().to(cfg.device)
        LOGGER.info("DONE.")

        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    elif cfg.tagger == "none":
        LOGGER.info("No tagging model will be used. Use SAM in dense sampling mode.")
        classes = ['item']
    else:
        raise ValueError(f"Unknown cfg.tagger: {cfg.tagger}")

    save_name = f"{cfg.tagger}"
    if cfg.get("exp_suffix"):
        save_name += f"_{cfg.exp_suffix}"

    global_classes: set[str] = set()
    obj_classes: set[str] = set()
    part_classes: set[str] = set()

    knowledge = load_knowledge(cfg)  # expects cfg.knowledge_path / skip_bg / bg_classes

    # ================= main loop =================
    for idx in trange(len(dataset)):
        color_path = Path(dataset.color_paths[idx])  # .../<scene>/rgb/frame_00000.jpg

        vis_save_path = color_path.parent.parent / cfg.save_folder_name / f"gsa_vis_{save_name}" / color_path.name
        detections_save_path = color_path.parent.parent / cfg.save_folder_name / f"gsa_detections_{save_name}" / color_path.name
        detections_save_path = detections_save_path.with_suffix(".pkl.gz")

        os.makedirs(os.path.dirname(str(vis_save_path)), exist_ok=True)
        os.makedirs(os.path.dirname(str(detections_save_path)), exist_ok=True)

        # OpenCV wants str paths
        image = cv2.imread(str(color_path))  # BGR
        if hasattr(dataset, 'camera_axis'):
            if dataset.camera_axis == 'Left':
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # ---------- tagging (RAM) ----------
        if cfg.detector in ("vlp", "gdino"):
            if cfg.tagger == "ram":
                raw_image = image_pil.resize((384, 384))
                raw_image = tagging_transform(raw_image).unsqueeze(0).to(cfg.device)
                ram_out = inference_ram(raw_image, tagging_model)[0]
                raw_tags = [t.strip() for t in ram_out.split(" | ")]

                obj_tags, part_tags = curate_tags(raw_tags, knowledge, cfg)
                obj_classes.update(obj_tags)
                part_classes.update(part_tags)
                global_classes.update(obj_tags + part_tags)

            if cfg.accumu_classes:
                classes = list(global_classes)
            else:
                classes = obj_tags + part_tags
        else:
            LOGGER.info("YOLO path does not use RAM tagging module.")

        # ---------- detection & segmentation ----------
        if cfg.tagger == "none":
            # SAM dense mode
            mask, xyxy, conf = get_sam_segmentation_dense(mask_generator, image_rgb)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)

            annotated_image, labels = vis_result_fast(image, detections, classes, instance_random_color=True)
            cv2.imwrite(str(vis_save_path), annotated_image)
            is_part = np.zeros((len(detections),), dtype=bool)
            is_obj  = np.zeros((len(detections),), dtype=bool)
            image_crops, image_feats, text_feats = [], [], []

        elif cfg.detector == "yolo":
            LOGGER.error("YOLO + SAM not implemented yet.")
            raise NotImplementedError

        elif cfg.detector == "gdino":
            LOGGER.error("GroundingDINO + SAM not implemented yet.")
            raise NotImplementedError

        elif cfg.detector == "vlp":
            # ---- VLPart detection ----
            detections = vlpart.predict_with_classes(image=image_rgb, classes=classes)

            # ---- NMS & clean-up ----
            if len(detections.class_id) > 0:
                valid = detections.class_id != -1
                if np.any(~valid):
                    sel0 = np.nonzero(valid)[0]
                    detections.xyxy       = detections.xyxy[sel0]
                    detections.confidence = detections.confidence[sel0]
                    detections.class_id   = detections.class_id[sel0]
                    if detections.mask is not None:
                        detections.mask = np.take(detections.mask, sel0, axis=0)
                    if detections.image_crops is not None:
                        detections.image_crops = [detections.image_crops[i] for i in sel0]
                    if detections.image_feats is not None:
                        detections.image_feats = np.take(detections.image_feats, sel0, axis=0)

                LOGGER.debug(f"Detections before NMS: {len(detections.class_id)} boxes")
                cls_names   = np.array([classes[cid] for cid in detections.class_id])
                is_part_m   = np.isin(cls_names, list(part_classes))
                is_obj_m    = np.isin(cls_names, list(obj_classes))
                is_unknown  = ~(is_part_m | is_obj_m)

                boxes  = torch.as_tensor(detections.xyxy, dtype=torch.float32)
                scores = torch.as_tensor(detections.confidence, dtype=torch.float32)

                keep_global = []
                if np.any(is_obj_m):
                    sub = np.flatnonzero(is_obj_m)
                    idx = torchvision.ops.nms(boxes[sub], scores[sub], float(cfg.nms_threshold))
                    keep_global.append(sub[idx.cpu().numpy()])
                if np.any(is_part_m):
                    sub = np.flatnonzero(is_part_m)
                    idx = torchvision.ops.nms(boxes[sub], scores[sub], float(cfg.part_nms_threshold))
                    keep_global.append(sub[idx.cpu().numpy()])
                if np.any(is_unknown):
                    sub = np.flatnonzero(is_unknown)
                    idx = torchvision.ops.nms(boxes[sub], scores[sub], float(cfg.nms_threshold))
                    keep_global.append(sub[idx.cpu().numpy()])
                    LOGGER.debug("Unknown detections after NMS.")

                if len(keep_global) == 0:
                    LOGGER.debug("No valid detections after NMS.")
                    keep_idx = np.empty((0,), dtype=int)
                else:
                    keep_idx = np.concatenate(keep_global, axis=0)

                # sync fields
                detections.xyxy       = detections.xyxy[keep_idx]
                detections.confidence = detections.confidence[keep_idx]
                detections.class_id   = detections.class_id[keep_idx]
                if detections.mask is not None:
                    detections.mask = np.take(detections.mask, keep_idx, axis=0)
                if detections.image_crops is not None:
                    detections.image_crops = [detections.image_crops[i] for i in keep_idx]
                if detections.image_feats is not None:
                    detections.image_feats = np.take(detections.image_feats, keep_idx, axis=0)

                # per-detection flags (aligned with final ordering)
                is_obj  = np.isin(np.array([classes[int(cid)] for cid in detections.class_id]), list(obj_classes)).astype(bool)
                is_part = np.isin(np.array([classes[int(cid)] for cid in detections.class_id]), list(part_classes)).astype(bool)

                LOGGER.debug(f"After NMS: {len(detections.xyxy)} boxes")

            else:
                is_part = np.zeros((0,), dtype=bool)
                is_obj  = np.zeros((0,), dtype=bool)

            # ---- SAM mask for kept boxes ----
            if len(detections.class_id) > 0:
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )
                image_crops = list(detections.image_crops) if detections.image_crops is not None else []
                image_feats = list(detections.image_feats) if detections.image_feats is not None else []
                # keep text_feats as class bank(C,D) + gather on demand
                if getattr(detections, "text_feats", None) is not None:
                    text_feats = list(detections.text_feats[detections.class_id])
                else:
                    text_feats = []
            else:
                image_crops, image_feats, text_feats = [], [], []

        else:
            raise ValueError(f"Unknown cfg.detector: {cfg.detector}")

        # ---------- visualize ----------
        annotated_image, labels = vis_result_fast(image, to_sv_detections(detections), classes)
        cv2.imwrite(str(vis_save_path), annotated_image)

        # ---------- save pickled detections ----------
        results = {
            "xyxy":        detections.xyxy,
            "confidence":  detections.confidence,
            "class_id":    detections.class_id,
            "mask":        detections.mask,
            "classes":     classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats":  text_feats, 
            "is_part":     is_part,
        }
        LOGGER.debug(f"\t= {int(np.sum(is_obj))} objects + {int(np.sum(is_part))} parts")

        if cfg.tagger == "ram":
            results["tagging_caption"] = "NA"
            results["tagging_text_prompt"] = raw_tags

        with gzip.open(str(detections_save_path), "wb") as f:
            pickle.dump(results, f)

    # save global classes
    classes_json = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / f"gsa_classes_{save_name}.json"
    os.makedirs(os.path.dirname(str(classes_json)), exist_ok=True)
    with open(str(classes_json), "w") as f:
        json.dump(list(global_classes), f)
    LOGGER.info(f"Saved classes to: {classes_json}")


# =========================
# Hydra entry
# =========================
@hydra.main(version_base=None, config_path="../configs", config_name="CAPA")  # [HYDRA]
def entry(cfg: DictConfig):
    _enrich_cfg_inplace(cfg)
    LOGGER.info("START main()")
    main(cfg)
    LOGGER.info("FINISH main()")

if __name__ == "__main__":
    entry()
    

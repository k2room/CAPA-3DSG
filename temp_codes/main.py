"""

"""
# === standard library ===
import os
import copy
from pathlib import Path
import pickle
import gzip
import json

# === third-party library ===
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
import supervision as sv
from collections import Counter
import logging

# === 2D open-vocab stack ===
from ultralytics import YOLO, SAM as ULTRA_SAM
import torchvision
import torchvision.transforms as TS

# === utils: conceptgraph.* ===
from src.utils.cg import (
    OptionalReRun, orr_log_annotated_image, orr_log_camera, orr_log_depth_image,
    orr_log_edges, orr_log_objs_pcd_and_bbox, orr_log_rgb_image, orr_log_vlm_image,
    OptionalWandB, DenoisingTracker, MappingTracker,
    consolidate_captions, get_obj_rel_from_image_gpt4v, get_openai_client,
    mask_subtract_contained,
    ObjectClasses, find_existing_image_path, get_det_out_path, get_exp_out_path,
    get_vlm_annotated_image_path, handle_rerun_saving, load_saved_detections,
    load_saved_hydra_json_config, make_vlm_edges_and_captions, measure_time,
    save_detection_results, save_edge_json, save_hydra_config, save_obj_json,
    save_objects_for_frame, save_pointcloud, should_exit_early, vis_render_image,
    # get_dataset,
    OnlineObjectRenderer, save_video_from_frames, vis_result_fast_on_depth,
    vis_result_for_vlm, vis_result_fast, save_video_detections,
    MapEdgeMapping, MapObjectList,
    filter_gobs, filter_objects, get_bounding_box, init_process_pcd,
    make_detection_list_from_pcd_and_gobs, denoise_objects, merge_objects,
    detections_to_obj_pcd_and_bbox, prepare_objects_save_vis, process_cfg,
    process_edges, process_pcd, processing_needed, resize_gobs,
    compute_spatial_similarities, compute_visual_similarities,
    aggregate_similarities, match_detections_to_objects, merge_obj_matches,
    compute_clip_features_batched, get_vis_out_path, cfg_to_dict, check_run_detections,
)
from src.utils.multiview import render_multiview_scene

# === utils: gsa functions ===
from src.utils.gsa import (
    GDINO, sam_model_registry, SamPredictor, RAM, inference_ram
)

from src.utils.vlp import VLPart

from src.utils.color_extraction import (
    extract_color_features,
    compute_texture_sim,
)
from src.utils.color_state import ColorFeatState

# === utils: utils functions, DynamicClasses ===
from src.utils.utils import (
    _seg_sam, _ensure_bool_mask, _vis_safe_det, load_knowledge, _curate_tags, 
    DynamicClasses
)

# === utils: dataloader ===
from src.dataloader.datasets_common import get_dataset

torch.set_grad_enabled(False)

# ============== detection branches (models preloaded) ==============

def run_yolo_branch(color_path: Path, obj_classes, cfg, yolo_model: YOLO, sam_pred: ULTRA_SAM, color_np):
    """
    Args:  
        color_path: Path to the color image
        obj_classes: ObjectClasses instance
        cfg: config
        yolo_model: preloaded YOLO model
        sam_pred: preloaded UltraSAM model
        color_np: numpy array of the color image (H, W, 3) uint8
    Returns: 
        det: sv.Detections instance
        labs: list of labels for each detection
        bgr: BGR image as numpy array (H, W, 3) uint8
    """
    results = yolo_model.predict(color_path, conf=0.1, verbose=False)
    conf = results[0].boxes.conf.cpu().numpy()
    cls_id = results[0].boxes.cls.cpu().numpy().astype(int)
    xyxy_t = results[0].boxes.xyxy
    xyxy = xyxy_t.cpu().numpy()

    if xyxy_t.numel() != 0:
        sam_out = sam_pred.predict(color_path, bboxes=xyxy_t, verbose=False)
        masks_t = sam_out[0].masks.data
        masks = _ensure_bool_mask(masks_t.cpu().numpy())
    else:
        masks = np.empty((0, *color_np.shape[:2]), dtype=bool)

    det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_id, mask=masks)
    labels = [f"{obj_classes.get_classes_arr()[cid]} {i}" for i, cid in enumerate(cls_id)]
    return det, labels

def run_ram_branch(color_path: Path, model: GDINO, sam_pred: SamPredictor, ram_pred: RAM, ram_tf, dyn: DynamicClasses, cfg, device, knowledge: dict):
    """
    Args:
        color_path: Path to the color image
        model: preloaded GDINO model
        sam_pred: preloaded SAM predictor (can be None when cfg.use_sam=False)
        ram_pred: preloaded RAM model
        ram_tf: RAM image transform
        dyn: DynamicClasses instance
        cfg: config
        device: torch device
        knowledge: object/part knowledge dict
    Returns:
        det2: sv.Detections (xyxy, confidence, class_id=global id, mask)
        labels: List[str]
        tags: List[str]
    """
    bgr = cv2.imread(str(color_path))

    # RAM -> tags
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((384, 384))
    inp = ram_tf(pil).unsqueeze(0).to(device)
    ram_out = inference_ram(inp, ram_pred)[0]
    tags = [t.strip() for t in ram_out.split(" | ") if t.strip()]

    # Apply object/part knowledge (remove/add/parts expansion)
    tags = _curate_tags(tags, knowledge, cfg)

    # add tags to dynamic classes
    dyn.add(tags)
    
    # DINO
    det = model.predict_with_classes(
        image=bgr, classes=tags,
        box_threshold=cfg.box_thresh, text_threshold=cfg.text_thresh
    )

    # NMS
    if len(det.xyxy) > 0 and getattr(cfg, "NMS_on", True):
        keep = torchvision.ops.nms(
            torch.from_numpy(det.xyxy),
            torch.from_numpy(det.confidence),
            cfg.NMS_iou_thresh
        ).cpu().numpy().tolist()
        det.xyxy = det.xyxy[keep]
        det.confidence = det.confidence[keep]
        det.class_id = det.class_id[keep]

    # SAM
    masks = _seg_sam(sam_pred, bgr, det.xyxy)

    gid = np.array([dyn.id_of(tags[c]) if len(tags) > 0 else -1 for c in det.class_id], dtype=int)

    det2 = sv.Detections(xyxy=det.xyxy, confidence=det.confidence, class_id=gid, mask=masks)
    gclasses = dyn.get_classes_arr()
    labels = []
    for i in range(len(det2.xyxy)):
        gi = int(det2.class_id[i])
        if 0 <= gi < len(gclasses):
            txt = gclasses[gi]
        else:
            txt = tags[int(det.class_id[i])] if len(tags) > 0 else "obj"
        labels.append(f"{txt} {float(det2.confidence[i]):0.2f}")

    return det2, labels, tags

def run_vlp_branch(color_path: Path, model: VLPart, sam_pred: SamPredictor, ram_pred: RAM, ram_tf, dyn: DynamicClasses, cfg, device, knowledge: dict):
    """
    Args:
        color_path: Path to the color image
        model: preloaded VLPart model
        sam_pred: preloaded SAM predictor (can be None when cfg.use_sam=False)
        ram_pred: preloaded RAM model
        ram_tf: RAM image transform
        dyn: DynamicClasses instance
        cfg: config
        device: torch device
        knowledge: object/part knowledge dict
    Returns:
        det2: sv.Detections (xyxy, confidence, class_id=global id, mask)
        labels: List[str]
        tags: List[str]
        image_crops, image_feats, text_feats  # from VLPart internal CLIP
        idx_obj: List[int]  # indices of detections considered objects
        idx_part: List[int] # indices of detections considered parts
    """
    bgr = cv2.imread(str(color_path))

    # RAM tags
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((384, 384))
    inp = ram_tf(pil).unsqueeze(0).to(device)
    ram_out = inference_ram(inp, ram_pred)[0]
    
    tags = [t.strip() for t in ram_out.split(" | ") if t.strip()]
    logging.debug(f"[2D Detection] RAM tags: {len(tags)} - {tags}")

    # Apply knowledge (remove/add/parts) and update dynamic classes
    tags = _curate_tags(tags, knowledge, cfg)
    dyn.add(tags)

    # VLPart inference with current vocabulary (uses internal CLIP)
    det = model.predict_with_classes(image=bgr, classes=tags)

    logging.debug(f"[2D Detection] class: {len(det.class_id)}, xyxy: {len(det.xyxy)}")
    logging.debug(det.class_id)
    # NMS (keep masks/features aligned)
    if len(det.xyxy) > 0 and getattr(cfg, "NMS_on", True):
        keep = torchvision.ops.nms(
            torch.from_numpy(det.xyxy),
            torch.from_numpy(det.confidence),
            cfg.NMS_iou_thresh
        ).cpu().numpy().tolist()
        det.xyxy = det.xyxy[keep]
        det.confidence = det.confidence[keep]
        det.class_id = det.class_id[keep]
        if getattr(det, "mask", None) is not None:
            det.mask = det.mask[keep]
        if getattr(det, "image_feats", None) is not None:
            det.image_feats = det.image_feats[keep]
        if getattr(det, "image_crops", None) is not None:
            det.image_crops = [det.image_crops[i] for i in keep]

    # Mask source: SAM toggle
    use_sam = bool(getattr(cfg, "use_sam", True))
    if not use_sam:
        masks = det.mask if getattr(det, "mask", None) is not None else None
        if masks is None:
            logging.debug("[VLPart] No mask detected.(use_sam=False)")
    else:
        if sam_pred is None:
            raise ValueError("cfg.use_sam=True but sam_pred is None. Check SAM loader.")
        masks = _seg_sam(sam_pred, bgr, det.xyxy)

    # Map to global class id
    gid = np.array([dyn.id_of(tags[c]) if len(tags) > 0 else -1 for c in det.class_id], dtype=int)
    det2 = sv.Detections(xyxy=det.xyxy, confidence=det.confidence, class_id=gid, mask=masks)
    logging.debug(f"[2D Detection] class: {len(det2.class_id)}, xyxy: {len(det2.xyxy)}")
    logging.debug(det.class_id)

    # Labels
    gclasses = dyn.get_classes_arr()
    labels = []
    for i in range(len(det2.xyxy)):
        gi = int(det2.class_id[i])
        txt = gclasses[gi] if 0 <= gi < len(gclasses) else (tags[int(det.class_id[i])] if len(tags) > 0 else "obj")
        labels.append(f"{txt} {float(det2.confidence[i]):0.2f}") # 'label confidence_score' form for visualization
    
    # Use VLPart-provided CLIP embeddings
    image_crops = getattr(det, "image_crops", None)
    image_feats = getattr(det, "image_feats", None)
    text_feats  = getattr(det, "text_feats", None)

    # ---- Split detections into object vs part buckets (by curated tags + knowledge) ----
    # Build part lexicon from knowledge
    parts_by_parent = knowledge.get('ram_add_part', {}) if isinstance(knowledge, dict) else {}
    part_tokens = {str(p).strip().lower() for parts in parts_by_parent.values() for p in parts}
    part_phrases = {f"{str(parent).strip().lower()} {str(p).strip().lower()}" for parent, parts in parts_by_parent.items() for p in parts}

    idx_obj, idx_part = [], []
    # Use local class ids from VLPart output to recover the tag for each detection
    if len(tags) == 0:
        idx_obj = list(range(len(det.xyxy)))
    else:
        for i, local_cid in enumerate(det.class_id):
            try:
                name = str(tags[int(local_cid)]).strip().lower()
            except Exception:
                name = ""
            toks = name.split()
            is_part = (name in part_phrases) or (len(toks) >= 2 and toks[-1] in part_tokens)
            (idx_part if is_part else idx_obj).append(i)

    return det2, labels, tags, image_crops, image_feats, text_feats, idx_obj, idx_part


def _split_obj_part_from_gobs(gobs: dict, knowledge: dict):
    """Split indices of current gobs into objects vs parts based on curated names.
    Uses knowledge['ram_add_part'] to detect part tokens.
    """
    parts_by_parent = knowledge.get('ram_add_part', {}) if isinstance(knowledge, dict) else {}
    part_tokens = {str(p).strip().lower() for parts in parts_by_parent.values() for p in parts}
    part_phrases = {f"{str(parent).strip().lower()} {str(p).strip().lower()}" for parent, parts in parts_by_parent.items() for p in parts}

    idx_obj, idx_part = [], []
    classes = gobs.get('classes', [])
    class_id = gobs.get('class_id', [])
    for i in range(len(gobs.get('xyxy', []))):
        try:
            name = str(classes[int(class_id[i])]).strip().lower()
        except Exception:
            name = ""
        toks = name.split()
        is_part = (name in part_phrases) or (len(toks) >= 2 and toks[-1] in part_tokens)
        (idx_part if is_part else idx_obj).append(i)
    return idx_obj, idx_part


# ============== main ==============
@hydra.main(version_base=None, config_path="/home/main/workspace/k2room2/CAPA-3DSG/configs/", config_name="main")
def main(cfg: DictConfig):
    # ==================== loggers ====================
    tracker = MappingTracker()
    logging.getLogger('PIL').setLevel(logging.INFO)
    logging.getLogger('Image').setLevel(logging.INFO)
    logging.getLogger('PngImagePlugin').setLevel(logging.INFO)

    if cfg.use_rerun:
        orr = OptionalReRun()
        orr.set_use_rerun(cfg.use_rerun)
        orr.init("realtime_mapping")
        orr.spawn()

    if cfg.use_wandb:
        owandb = OptionalWandB()
        owandb.set_use_wandb(cfg.use_wandb)
        owandb.init(project="concept-graphs", config=cfg_to_dict(cfg))

    # ==================== setting ====================
    cfg = process_cfg(cfg)

    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(view_param=view_param, base_objects=None, gray_map=False)
        frames = []

    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    if getattr(cfg, "out_dir", ""):
        exp_out_path = Path(cfg.out_dir)
        det_exp_path = exp_out_path / "detections"
        exp_out_path.mkdir(parents=True, exist_ok=True)

    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    prev_adjusted_pose = None

    # ==================== dataset ====================
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start, end=cfg.end, stride=cfg.stride,
        basedir=cfg.dataset_root, sequence=cfg.scene_id,
        desired_height=cfg.image_height, desired_width=cfg.image_width,
        device="cpu", dtype=torch.float,
    )

    # ==================== graph ====================
    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)
    knowledge = load_knowledge(cfg)
    color_state = ColorFeatState(ema_alpha=float(cfg.get('color_ema_alpha', 0.30)))

    # ==================== detection models ====================
    yolo_model = None
    clip_model = clip_preprocess = clip_tokenizer = None
    obj_classes = None
    sam_pred = None  

    if run_detections:
        print("LOAD DETECTION MODELS...")
        if cfg.detector_mode == "YOLO" or cfg.detector_mode == "GDINO":
            print("LOAD CLIP MODELS...")
            det_exp_path.mkdir(parents=True, exist_ok=True)
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
            clip_model = clip_model.to(cfg.device)
            clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            print("CLIP MODELS LOADED.")
        
        if cfg.detector_mode == "YOLO":
            print("LOAD YOLO + SAM MODELS...")
            detections_exp_cfg = cfg_to_dict(cfg)
            base_classes_file = Path(detections_exp_cfg['classes_file'])
            obj_classes = ObjectClasses(
                classes_file_path=str(base_classes_file),
                bg_classes=detections_exp_cfg['bg_classes'],
                skip_bg=detections_exp_cfg['skip_bg']
            )
            yolo_model = YOLO('yolov8l-world.pt')
            yolo_model.set_classes(obj_classes.get_classes_arr())
            sam_pred = ULTRA_SAM('sam_l.pt')
            print("YOLO + SAM MODELS LOADED.")

        elif cfg.detector_mode == "GDINO":
            print("LOAD RAM + GDINO + SAM MODELS...")
            obj_classes = DynamicClasses(
                bg_classes=getattr(cfg, "bg_classes", []),
                skip_bg=getattr(cfg, "skip_bg", False),
                colors_file_path=None,
                rng_seed=0
            )
            gdino = GDINO(model_config_path=cfg.gdino_cfg_path, model_checkpoint_path=cfg.gdino_ckpt_path)
            sam_pred = SamPredictor(sam_model_registry[cfg.sam_encoder](checkpoint=cfg.sam_ckpt_path))
            ram_pred = RAM(pretrained=cfg.ram_ckpt_path, image_size=384, vit='swin_l').to(cfg.device)
            ram_pred.eval()
            ram_tf = TS.Compose([TS.Resize((384,384)), TS.ToTensor(), TS.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
            print("RAM + GDINO + SAM MODELS LOADED.")

        elif cfg.detector_mode == "CAPA":
            print("LOAD RAM + VLPart + SAM MODELS...")
            obj_classes = DynamicClasses(
                bg_classes=getattr(cfg, "bg_classes", []),
                skip_bg=getattr(cfg, "skip_bg", False),
                colors_file_path=None,
                rng_seed=0
            )
            vlpart = VLPart(model_path=cfg.vlpart_ckpt_path, config_file=cfg.vlpart_cfg_path, device=cfg.device, score_thresh=cfg.score_thresh)
            if getattr(cfg, "use_sam", True):
                sam_pred = SamPredictor(sam_model_registry[cfg.sam_encoder](checkpoint=cfg.sam_ckpt_path))
            ram_pred = RAM(pretrained=cfg.ram_ckpt_path, image_size=384, vit='swin_l').to(cfg.device)
            ram_pred.eval()
            ram_tf = TS.Compose([TS.Resize((384,384)), TS.ToTensor(), TS.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
            print("RAM + VLPart + SAM MODELS LOADED.")

        else:
            raise NotImplementedError(f"Unknown detector_mode: {cfg.detector_mode}")

    # ==================== save config ====================
    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(cfg_to_dict(cfg), exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)
    
    # ==================== main loop ====================
    exit_early_flag = False
    counter = 0
    for frame_idx in trange(len(dataset)):
        if cfg.max_frames != 0 and frame_idx >= cfg.max_frames:
            print("Stop for Debug: max_frames reached.")
            break

        tracker.curr_frame_idx = frame_idx
        counter += 1
        if cfg.use_rerun:
            orr.set_time_sequence("frame", frame_idx)

        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # ==== Sanity checks ====
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy()
        image_rgb = (color_np).astype(np.uint8) 
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"


        raw_gobs = None
        gobs = None
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        
        # vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        # vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)

        if run_detections:
            # ==== Load frame ====
            image = cv2.imread(str(color_path)) # BGR uint8
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_feats = None  # default for non-CAPA modes

            if cfg.detector_mode == "YOLO":
                curr_det, det_labels = measure_time(run_yolo_branch)(
                    color_path=color_path, obj_classes=obj_classes, cfg=cfg,
                    yolo_model=yolo_model, sam_pred=sam_pred, color_np=color_np
                )
                image_crops, image_feats, text_feats = compute_clip_features_batched(
                    image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer,
                    obj_classes.get_classes_arr(), cfg.device
                )

            elif cfg.detector_mode == "GDINO":
                curr_det, det_labels, tags = measure_time(run_ram_branch)(
                    color_path=color_path, model=gdino, sam_pred=sam_pred,
                    ram_pred=ram_pred, ram_tf=ram_tf,
                    dyn=obj_classes, cfg=cfg, device=cfg.device, knowledge=knowledge
                )
                image_crops, image_feats, text_feats = compute_clip_features_batched(
                    image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer,
                    obj_classes.get_classes_arr(), cfg.device
                )

            elif cfg.detector_mode == "CAPA":
                curr_det, det_labels, tags, image_crops, image_feats, text_feats, idx_obj, idx_part = measure_time(run_vlp_branch)(
                    color_path=color_path, model=vlpart, sam_pred=sam_pred,
                    ram_pred=ram_pred, ram_tf=ram_tf,
                    dyn=obj_classes, cfg=cfg, device=cfg.device, knowledge=knowledge
                )
                
                color_params = dict(
                    use_wb=bool(cfg.get('color_use_wb', True)),
                    use_retinex=bool(cfg.get('color_use_retinex', False)),
                    bins=int(cfg.get('color_bins', 32)),
                    s_threshold=float(cfg.get('color_s_threshold', 0.10)),
                    v_spec_threshold=float(cfg.get('color_v_spec_threshold', 0.90)),
                    compute_cn=bool(cfg.get('color_compute_cn', True)),
                )
                # Compute color features only for part masks; keep objects as None
                if getattr(curr_det, 'mask', None) is not None and len(idx_part) > 0:
                    N = len(curr_det.xyxy)
                    color_feats = [None] * N
                    part_idx_arr = np.asarray(idx_part, dtype=int)
                    feats_part = measure_time(extract_color_features)(image_rgb, curr_det.mask[part_idx_arr], params=color_params)
                    for k, mi in enumerate(idx_part):
                        color_feats[mi] = feats_part[k]
                else:
                    color_feats = [None] * (len(curr_det.xyxy) if getattr(curr_det, 'xyxy', None) is not None else 0)

            else:
                raise NotImplementedError(f"Unknown detector_mode: {cfg.detector_mode}")

            # labels_vlm, edges, edge_image, captions = make_vlm_edges_and_captions(
            #     bgr, curr_det, obj_classes, det_labels, det_exp_vis_path, color_path, False, None
            # )image
            labels_vlm, edges, edge_image, captions = [], [], None, []

            tracker.increment_total_detections(len(curr_det.xyxy))
            results = {
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": det_labels,
                "labels": labels_vlm,
                "edges": edges,
                "captions": captions,
                "color_feats": color_feats,
            }

            # ===== save detections =====
            if cfg.save_detections:
                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                box_annot = sv.BoxAnnotator()
                mask_annot = sv.MaskAnnotator()
                viz_det = _vis_safe_det(curr_det)

                ann_img = mask_annot.annotate(scene=image.copy(), detections=viz_det)
                ann_img = box_annot.annotate(scene=ann_img, detections=viz_det, labels=det_labels)
                cv2.imwrite(str(vis_save_path), ann_img)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                ann_depth = mask_annot.annotate(scene=depth_image_rgb.copy(), detections=viz_det)
                ann_depth = box_annot.annotate(scene=ann_depth, detections=viz_det, labels=det_labels)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), ann_depth)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)

                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)

            raw_gobs = results

        # ===== load old saving detections =====
        else:
            stem = Path(dataset.color_paths[frame_idx]).stem # Support current and old saving formats
            if os.path.exists(det_exp_pkl_path / stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(stem):06}")
            else:
                # if no detections, throw an error
                raise FileNotFoundError(
                    f"No detections found for frame {frame_idx} at paths \n"
                    f"{det_exp_pkl_path / stem} or \n"
                    f"{det_exp_pkl_path / f'{int(stem):06}'}."
                )

        # ===== pose/cam logs =====
        unt_pose = dataset.poses[frame_idx].cpu().numpy() # untrasformed pose
        adjusted_pose = unt_pose

        if cfg.use_rerun:
            prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.image_width, cfg.image_height, frame_idx)
            orr_log_rgb_image(color_path)
            orr_log_annotated_image(color_path, det_exp_vis_path)
            orr_log_depth_image(depth_tensor)
            orr_log_vlm_image(vis_save_path_for_vlm)
            orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

        # ===== filtering =====
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        filtered_out = filter_gobs(
            resized_gobs, image_rgb,
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
            return_index_map=True
        )

        # gobs = filtered_gobs
        # Unpack (gobs, raw->filt index map). Backward compatible if tuple not returned.
        if isinstance(filtered_out, tuple):
            filtered_gobs, idx_map_raw2filt = filtered_out
        else:
            filtered_gobs, idx_map_raw2filt = filtered_out, None
        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            continue

        # ===== seperate the overlapped masks (so that each mask only covers one object) =====
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        # ==================== Mask to Point Cloud ====================
        if cfg.detector_mode == "CAPA":
            # Use index map to update indices from run_vlp_branch() without recomputing
            if 'idx_map_raw2filt' in locals() and idx_map_raw2filt is not None:
                idx_obj_filt  = [idx_map_raw2filt[i] for i in idx_obj  if i in idx_map_raw2filt]
                idx_part_filt = [idx_map_raw2filt[i] for i in idx_part if i in idx_map_raw2filt]
            #     try:
            #         idx_obj_filt  = [idx_map_raw2filt[i] for i in idx_obj  if i in idx_map_raw2filt]
            #         idx_part_filt = [idx_map_raw2filt[i] for i in idx_part if i in idx_map_raw2filt]
            #     except NameError:
            #         # Fallback (e.g., when loading cached detections w/o idx_obj/idx_part)
            #         idx_obj_filt, idx_part_filt = _split_obj_part_from_gobs(gobs, knowledge)
            # else:
            #     # No map available -> fallback to name-based split
            #     idx_obj_filt, idx_part_filt = _split_obj_part_from_gobs(gobs, knowledge)

            obj_pcds_and_bboxes = [None] * len(gobs['mask'])
            K = intrinsics.cpu().numpy()[:3, :3]

            # Objects: downsample to target points (default cfg.obj_pcd_max_points)
            if len(idx_obj_filt) > 0:
                masks_obj = gobs['mask'][np.asarray(idx_obj_filt, dtype=int)]
                target_obj = int(getattr(cfg, 'obj_pcd_max_points_obj', getattr(cfg, 'obj_pcd_max_points', 5000)))
                obj_list = measure_time(detections_to_obj_pcd_and_bbox)(
                    depth_array=depth_array,
                    masks=masks_obj,
                    cam_K=K,
                    image_rgb=image_rgb,
                    trans_pose=adjusted_pose,
                    min_points_threshold=cfg.min_points_threshold,
                    spatial_sim_type=cfg.spatial_sim_type,
                    obj_pcd_max_points=target_obj,
                    device=cfg.device,
                )
                for k, mi in enumerate(idx_obj_filt):
                    obj_pcds_and_bboxes[mi] = obj_list[k]

            # Parts: bypass downsampling with target=-1
            if len(idx_part_filt) > 0:
                masks_part = gobs['mask'][np.asarray(idx_part_filt, dtype=int)]
                target_part = int(getattr(cfg, 'obj_pcd_max_points_part', -1))
                part_list = measure_time(detections_to_obj_pcd_and_bbox)(
                    depth_array=depth_array,
                    masks=masks_part,
                    cam_K=K,
                    image_rgb=image_rgb,
                    trans_pose=adjusted_pose,
                    min_points_threshold=cfg.min_points_threshold,
                    spatial_sim_type=cfg.spatial_sim_type,
                    obj_pcd_max_points=target_part,
                    device=cfg.device,
                )
                for k, mi in enumerate(idx_part_filt):
                    obj_pcds_and_bboxes[mi] = part_list[k]

            # ==================== remove noise ====================
            for i, obj in enumerate(obj_pcds_and_bboxes):
                if obj:
                    if i in idx_part_filt:
                        # parts: keep dense points; optionally skip DBSCAN/downsample
                        pass
                    else:
                        obj["pcd"] = init_process_pcd(
                            pcd=obj["pcd"],
                            downsample_voxel_size=cfg["downsample_voxel_size"],
                            dbscan_remove_noise=cfg["dbscan_remove_noise"],
                            dbscan_eps=cfg["dbscan_eps"],
                            dbscan_min_points=cfg["dbscan_min_points"],
                        )
                    obj["bbox"] = get_bounding_box(spatial_sim_type=cfg['spatial_sim_type'], pcd=obj["pcd"])
        else:
            obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
                depth_array=depth_array,
                masks=gobs['mask'],
                cam_K=intrinsics.cpu().numpy()[:3, :3],
                image_rgb=image_rgb,
                trans_pose=adjusted_pose,
                min_points_threshold=cfg.min_points_threshold,
                spatial_sim_type=cfg.spatial_sim_type,
                obj_pcd_max_points=cfg.obj_pcd_max_points,
                device=cfg.device,
            )

            # ==================== remove noise ====================
            for obj in obj_pcds_and_bboxes:
                if obj:
                    obj["pcd"] = init_process_pcd(
                        pcd=obj["pcd"],
                        downsample_voxel_size=cfg["downsample_voxel_size"],
                        dbscan_remove_noise=cfg["dbscan_remove_noise"],
                        dbscan_eps=cfg["dbscan_eps"],
                        dbscan_min_points=cfg["dbscan_min_points"],
                    )
                    obj["bbox"] = get_bounding_box(spatial_sim_type=cfg['spatial_sim_type'], pcd=obj["pcd"])

        # Build detection list; in CAPA we pass object/part indices for downstream weighting
        if cfg.detector_mode == "CAPA":
            detection_list = make_detection_list_from_pcd_and_gobs(
                obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx,
                idx_obj_filt=idx_obj_filt, idx_part_filt=idx_part_filt,
            )
        else:
            detection_list = make_detection_list_from_pcd_and_gobs(
                obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
            )
        if len(detection_list) == 0: # no detections in this frame
            continue

        # if no objects yet in the map, just add all the objects from the current frame (no need to match or merge)
        if len(objects) == 0:
            objects.extend(detection_list)
            # Seed color features for newly added objects from current detections
            if cfg.detector_mode == "CAPA" and 'color_feats' in gobs:
                color_state.seed_from_detections(detection_list, gobs['color_feats'])
            tracker.increment_total_objects(len(detection_list))
            if cfg.use_wandb:
                owandb.log({"total_objects_so_far": tracker.get_total_objects(), "objects_this_frame": len(detection_list)})
            continue

        # ==================== calculate similarity, match and merge ====================
        if cfg.detector_mode == "YOLO" or cfg.detector_mode == "GDINO":
            # Geometric similarity
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=cfg['spatial_sim_type'],
                detection_list=detection_list, objects=objects,
                downsample_voxel_size=cfg['downsample_voxel_size']
            )

            # Semantic similarity
            visual_sim = compute_visual_similarities(detection_list, objects)

            agg_sim = aggregate_similarities(
                match_method=cfg['match_method'], phys_bias=cfg['phys_bias'],
                spatial_sim=spatial_sim, visual_sim=visual_sim
            )

        elif cfg.detector_mode == "CAPA":
            # Geometric similarity
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=cfg['spatial_sim_type'],
                detection_list=detection_list, objects=objects,
                downsample_voxel_size=cfg['downsample_voxel_size']
            )

            # Semantic similarity
            visual_sim = compute_visual_similarities(detection_list, objects)

            # Texture similarity (aligned to detection_list by mask_idx), only for parts
            M = len(detection_list)
            if 'color_feats' in gobs:
                det_color_feats = []
                for det in detection_list:
                    try:
                        mi = det.get('mask_idx', [None])[0]
                        det_color_feats.append(gobs['color_feats'][mi] if mi is not None else None)
                    except Exception:
                        det_color_feats.append(None)
            else:
                det_color_feats = [None] * M
            obj_color_feats = color_state.get_obj_feat_list(objects)
            texture_sim = compute_texture_sim(
                det_color_feats,
                obj_color_feats,
                weights=tuple(cfg.get('color_ab_rg_cn_weights', [0.50, 0.20, 0.20, 0.10])),
                mapping=cfg.get('color_sim_mapping', 'inv'),    # 'inv' or 'exp'
                gamma=float(cfg.get('color_sim_gamma', 3.0)),
            )

            # Weighted aggregation with part/object separation
            phys_bias = float(cfg.get('phys_bias', 0.0))
            beta = float(cfg.get('texture_weight', 0.30))
            w_s_default = 1.0 + phys_bias
            w_v_default = (1.0 - phys_bias) * (1.0 - beta)
            w_t_default = (1.0 - phys_bias) * beta
            w_s = float(cfg.get('sim_w_spatial', w_s_default))
            w_v = float(cfg.get('sim_w_visual', w_v_default))
            w_t = float(cfg.get('sim_w_texture', w_t_default))

            # Per-detection type mask
            part_mask = torch.tensor([1.0 if d.get('is_part', False) else 0.0 for d in detection_list])
            part_mask = part_mask.view(-1, 1)  # (M,1)

            agg_sim_obj = w_s * spatial_sim + w_v * visual_sim
            agg_sim_part = agg_sim_obj + w_t * texture_sim
            agg_sim = (1.0 - part_mask) * agg_sim_obj + part_mask * agg_sim_part

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(agg_sim=agg_sim, detection_threshold=cfg['sim_threshold'])

        pre_len_objects = len(objects)

        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
        )

        # Post-merge: update/assign object color features via EMA in sidecar map (no mutations to thirdparty objects)
        if cfg.detector_mode == "CAPA":
            color_state.update_post_merge(objects, match_indices, det_color_feats, pre_len_objects)

        # voting to set majority class (ignore negative)
        for idx, obj in enumerate(objects):
            valid_ids = [i for i in obj['class_id'] if i >= 0]
            if not valid_ids:
                continue
            most_id = Counter(valid_ids).most_common(1)[0][0]
            cname = obj_classes.get_classes_arr()[most_id]
            if obj["class_name"] != cname:
                obj["class_name"] = cname

        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        # Clean up outlier edges
        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            obj1_class_name = objects[curr_obj1_idx]['class_name'] 
            obj2_class_name = objects[curr_obj2_idx]['class_name']
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections
            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])

        # ==================== post-processing (denoise_objects, filter_objects, merge_objects) ====================
        is_final_frame = frame_idx == len(dataset) - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # light cleanup / post steps (unchanged) for every process_interval or final frame
        if processing_needed(cfg["denoise_interval"], cfg["run_denoise_final_frame"], frame_idx, is_final_frame):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'],
                dbscan_remove_noise=cfg['dbscan_remove_noise'],
                dbscan_eps=cfg['dbscan_eps'], dbscan_min_points=cfg['dbscan_min_points'],
                spatial_sim_type=cfg['spatial_sim_type'], device=cfg['device'], objects=objects
            )
        if processing_needed(cfg["filter_interval"], cfg["run_filter_final_frame"], frame_idx, is_final_frame):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], obj_min_detections=cfg['obj_min_detections'],
                objects=objects, map_edges=map_edges
            )
        if processing_needed(cfg["merge_interval"], cfg["run_merge_final_frame"], frame_idx, is_final_frame):
            # Ensure no stray custom keys exist before calling thirdparty merge
            if cfg.detector_mode == "CAPA":
                for obj in objects:
                    if isinstance(obj, dict) and 'color_feat' in obj:
                        try:
                            del obj['color_feat']
                        except Exception:
                            pass
            objects, map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects, downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"], dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"], spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"], do_edges=cfg["make_edges"], map_edges=map_edges
            )

        # ==================== saving and visualization ====================
        if cfg.use_rerun:
            orr_log_objs_pcd_and_bbox(objects, obj_classes)
            orr_log_edges(objects, map_edges, obj_classes)

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}",
                frame_idx, objects, cfg.obj_min_detections, adjusted_pose, color_path
            )

        if cfg.vis_render:
            vis_render_image(
                objects, obj_classes, obj_renderer, image_original_pil, adjusted_pose, None, frame_idx,
                color_path, cfg.obj_min_detections, cfg.class_agnostic, cfg.debug_render,
                is_final_frame, cfg.exp_out_path, cfg.exp_suffix,
            )

        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            save_pointcloud(
                exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path, cfg=cfg,
                objects=objects, obj_classes=obj_classes, latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True
            )

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))

        if cfg.use_wandb:
            owandb.log({
                "frame_idx": frame_idx, "counter": counter,
                "is_final_frame": is_final_frame,
            })
        
        if cfg.use_wandb:
            owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
            })
    # ===================== end of main loop ====================

    # captions
    for obj in objects:
        caps = obj['captions'][:20]
        obj['consolidated_caption'] = caps

    if cfg.use_rerun:
        handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)

    # ===================== save dynamic classes =====================
    if run_detections and cfg.detector_mode == "GDINO":
        out_cls = exp_out_path / getattr(cfg, "classes_output_filename", "ram_classes.txt")
        obj_classes.save(out_cls)
        print(f"[RAM] saved classes -> {out_cls}")
    
    # ==================== final saving and visualization ====================
    if run_detections and cfg.save_video:
        save_video_detections(det_exp_path)

    if cfg.save_json:
        save_obj_json(cfg.exp_suffix, exp_out_path, objects)
        save_edge_json(cfg.exp_suffix, exp_out_path, objects, map_edges)

    if cfg.save_objects_all_frames:
        save_meta_path = (exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}" / "meta.pkl.gz")
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': {},  # optional: can be filled if needed
            }, f)
    
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path, cfg=cfg,
            objects=objects, obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath, create_symlink=True,
            edges=map_edges
        )

        pcd_save_path = Path(exp_out_path) / f"pcd_{cfg.exp_suffix}.pkl.gz"
        print(f"\nUse '$ python src/utils/visualize_saved_pc.py --pcd_path {pcd_save_path}' to visualize the point cloud.")

    # ===================== rendering multiview snapshots =====================
    if bool(cfg.get('render_multiview', True)):
        try:
            color_mode = cfg.get('multiview_color_mode', 'rgb')
            image_size=tuple(cfg.get('multiview_image_size', [800, 600]))
            fov_deg=cfg.get('multiview_fov_deg', 120.0)
            radius_scale=float(cfg.get('multiview_radius_scale', 1.3))
            point_size=float(cfg.get('multiview_point_size', 3.5))
            include_bboxes=bool(cfg.get('multiview_include_bboxes', False))
            
            render_multiview_scene(
                objects=objects,
                out_dir=exp_out_path,
                obj_min_detections=cfg.obj_min_detections,
                exclude_background=False,
                image_size=image_size,
                fov_deg=fov_deg,
                radius_scale=radius_scale,
                point_size=point_size,
                include_bboxes=include_bboxes,
                color_mode=color_mode,
                obj_classes=(obj_classes if color_mode == 'class' else None),
            )
            print(f"Saved multiview snapshots to: {exp_out_path / 'multiview'}")
        except Exception as e:
            print(f"[Multiview] Failed to render snapshots: {e}")

    if cfg.use_wandb:
        owandb.finish()

if __name__ == "__main__":
    main()

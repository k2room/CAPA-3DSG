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

# === utils: gsa functions ===
from src.utils.gsa import (
    GDINO, sam_model_registry, SamPredictor, RAM, inference_ram
)

from src.utils.vlp import VLPart

# === utils: utils functions, DynamicClasses ===
from src.utils.utils import (
    # _build_OFG_stack, 
    _seg_sam, _ensure_bool_mask, _vis_safe_det, load_knowledge, _curate_tags, 
    DynamicClasses
)

# === utils: dataloader ===
from src.dataloader.datasets_common import get_dataset

torch.set_grad_enabled(False)

# ============== detection branches (models preloaded) ==============

def run_yolo_branch(color_path: Path, obj_classes, cfg, yolo_model: YOLO, usam: ULTRA_SAM, color_np):
    """
        Args:  
            color_path: Path to the color image
            obj_classes: ObjectClasses instance
            cfg: config
            yolo_model: preloaded YOLO model
            usam: preloaded UltraSAM model
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
        sam_out = usam.predict(color_path, bboxes=xyxy_t, verbose=False)
        masks_t = sam_out[0].masks.data
        masks = _ensure_bool_mask(masks_t.cpu().numpy())
    else:
        masks = np.empty((0, *color_np.shape[:2]), dtype=bool)

    det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_id, mask=masks)
    labels = [f"{obj_classes.get_classes_arr()[cid]} {i}" for i, cid in enumerate(cls_id)]
    return det, labels

def run_ram_branch(color_path: Path, model: GDINO, sam_pred: SamPredictor, ram_pred: RAM, ram_tf, dyn: DynamicClasses, cfg, device, knowledge: dict):
    bgr = cv2.imread(str(color_path))

    # RAM -> tags
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((384, 384))
    inp = ram_tf(pil).unsqueeze(0).to(device)
    ram_out = inference_ram(inp, ram_pred)  # "a | b | c"
    tags = [t.strip() for t in ram_out[0].split(" | ") if t.strip()]

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
            cfg.iou_thresh
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

    # ==================== detection models ====================
    yolo_model = usam = None
    clip_model = clip_preprocess = clip_tokenizer = None
    obj_classes = None  

    if run_detections:
        print("LOAD DETECTION MODELS...")
        if cfg.detector_mode == "CG" or cfg.detector_mode == "OFG":
            print("LOAD CLIP MODELS...")
            det_exp_path.mkdir(parents=True, exist_ok=True)
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
            clip_model = clip_model.to(cfg.device)
            clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            print("CLIP MODELS LOADED.")
        
        if cfg.detector_mode == "CG":
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
            usam = ULTRA_SAM('sam_l.pt')
            print("YOLO + SAM MODELS LOADED.")

        elif cfg.detector_mode == "OFG":
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
            vlpart = VLPart(model_path=cfg.vlpart_ckpt_path, config_file=cfg.vlpart_cfg_path, device=cfg.device, score_thresh=cfg.box_thresh)
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
    return 
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

            if cfg.detector_mode == "CG":
                curr_det, det_labels = measure_time(run_yolo_branch)(
                    color_path=color_path, obj_classes=obj_classes, cfg=cfg,
                    yolo_model=yolo_model, usam=usam, color_np=color_np
                )
                image_crops, image_feats, text_feats = compute_clip_features_batched(
                    image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer,
                    obj_classes.get_classes_arr(), cfg.device
                )

            elif cfg.detector_mode == "OFG":
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
                curr_det, det_labels, tags = measure_time(run_ram_branch)(
                    color_path=color_path, model=vlpart, sam_pred=sam_pred,
                    ram_pred=ram_pred, ram_tf=ram_tf,
                    dyn=obj_classes, cfg=cfg, device=cfg.device, knowledge=knowledge
                )

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
        filtered_gobs = filter_gobs(
            resized_gobs, image_rgb,
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            continue

        # ===== seperate the overlapped masks (so that each mask only covers one object) =====
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        # ==================== Mask to Point Cloud ====================
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

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )
        if len(detection_list) == 0: # no detections in this frame
            continue

        # if no objects yet in the map, just add all the objects from the current frame (no need to match or merge)
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            if cfg.use_wandb:
                owandb.log({"total_objects_so_far": tracker.get_total_objects(), "objects_this_frame": len(detection_list)})
            continue

        # ==================== match and merge ====================
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'],
            detection_list=detection_list, objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )
        visual_sim = compute_visual_similarities(detection_list, objects)
        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], phys_bias=cfg['phys_bias'],
            spatial_sim=spatial_sim, visual_sim=visual_sim
        )
        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(agg_sim=agg_sim, detection_threshold=cfg['sim_threshold'])
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
    if run_detections and cfg.detector_mode == "OFG":
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

    if cfg.use_wandb:
        owandb.finish()

if __name__ == "__main__":
    main()

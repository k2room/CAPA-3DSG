"""
3D Fusion using the detected objects/parts from each frame.
- use CAPA_slam.yaml for configuration (Hydra).
- example:
    $ python scripts/3D_fusion.py scene_id=0kitchen/video0 dataset=FunGraph3D save_folder_name=capa_wc_1 mask_conf_threshold=0.30 max_bbox_area_ratio=0.90 merge_overlap_thresh=0.2 merge_visual_sim_thresh=0.6 merge_text_sim_thresh=0.8
    $ python scripts/3D_fusion.py scene_id=0kitchen/video0 dataset=FunGraph3D save_folder_name=capa_wc_1 mask_conf_threshold=0.15 max_bbox_area_ratio=0.15 merge_overlap_thresh=0.5 merge_visual_sim_thresh=0.75 merge_text_sim_thresh=0.7 part_reg=True
"""
# ===== imports =====
import copy
from datetime import datetime
import os
from pathlib import Path
import gzip
import pickle
import numpy as np
import open3d as o3d
import torch
from tqdm import trange
import warnings, logging

# ===== hydra / omegaconf =====
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

# register custom resolver used in logging filename: ${replace:x,/,-}
OmegaConf.register_new_resolver("replace", lambda s, a, b: str(s).replace(a, b))

warnings.simplefilter(action='ignore', category=FutureWarning)

# ===== project imports =====
from dataloader.datasets_common import get_dataset
from utils.vis import OnlineObjectRenderer
from utils.ious import compute_2d_box_contained_batch
from utils.color_extraction import compute_texture_sim 
from slam.slam_classes import MapObjectList
from slam.utils import (
    create_or_load_colors,
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
)
from slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)   

# Disable torch gradient computation
torch.set_grad_enabled(False)

LOGGER = logging.getLogger(__name__)  # [HYDRA] use hydra-managed logger
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('Image').setLevel(logging.INFO)
logging.getLogger('PngImagePlugin').setLevel(logging.INFO)

# =========================
# Config enrichment
# =========================
def _resolve_path(p) -> str:
    """Resolve path against original CWD (Hydra changes run dir)."""
    pp = Path(str(p))
    if not pp.is_absolute():
        pp = Path(get_original_cwd()) / pp
    return str(pp)

def _process_cfg(cfg: DictConfig) -> None:  # [HYDRA]
    """
    - require scene_id & dataset
    - attach dataset_root / dataset_config
    """
    if not cfg.get("scene_id") or not cfg.get("dataset"):
        raise ValueError("Both `scene_id` and `dataset` are required. e.g., scene_id=0kitchen/video0 dataset=FunGraph3D")
    if str(cfg.dataset) not in cfg.ALLOWED_DATASETS:
        raise ValueError(f"`dataset` must be one of {sorted(cfg.ALLOWED_DATASETS)}; got {cfg.dataset}")

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
    
    dataset_cfg = OmegaConf.load(cfg.dataset_config)
    if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
    if cfg.image_width is None:
        cfg.image_width = dataset_cfg.camera_params.image_width

    OmegaConf.set_struct(cfg, prev_struct)

def compute_match_batch(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Compute object association based on spatial and visual similarities
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of binary values, indicating whether a detection is associate with an object. 
        Each row has at most one 1, indicating one detection can be associated with at most one existing object.
        One existing object can receive multiple new detections
    '''
    assign_mat = torch.zeros_like(spatial_sim)
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
        row_max, row_argmax = torch.max(sims, dim=1) # (M,), (M,)
        for i in row_max.argsort(descending=True):
            if row_max[i] > cfg.sim_threshold:
                assign_mat[i, row_argmax[i]] = 1
            else:
                break
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return assign_mat

def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float=0.01):
    objects_to_save = copy.deepcopy(objects)
            
    # Downsample the point cloud
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # Remove unnecessary keys
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in [
                'pcd', 'bbox', 'clip_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color'
            ]:
                del objects_to_save[i][k]
                
    return objects_to_save.to_serializable()

def compute_color_similarities(cfg, detection_list, objects) -> torch.Tensor:
    """
    Compute color similarity matrix (M x N) from per-detection and per-object `color_feat` dicts.
    - detection_list[i]['color_feat'] : Optional[Dict]
    - objects[j]['color_feat']       : Optional[Dict]
    Returns torch.FloatTensor on cfg.device.
    """
    M, N = len(detection_list), len(objects)
    if M == 0 or N == 0:
        return torch.zeros((M, N), dtype=torch.float32, device=cfg.device)

    det_feats = [detection_list[i].get('color_feat', None) for i in range(M)]
    obj_feats = [objects[j].get('color_feat', None)       for j in range(N)]

    weights = getattr(cfg, "color_distance_weights", None)      # dict: {'ab','opp','cn','rg','med'}
    mapping = getattr(cfg, "color_sim_mapping", "inv")          # 'inv' or 'exp'
    gamma   = float(getattr(cfg, "color_sim_gamma", 3.0))       # for 'exp'

    S = compute_texture_sim(det_feats, obj_feats, weights=weights, mapping=mapping, gamma=gamma)
    if not torch.is_tensor(S):
        S = torch.tensor(S, dtype=torch.float32)
    return S.to(device=cfg.device, dtype=torch.float32)

def aggregate_similarities_wc(
    cfg,
    spatial_sim: torch.Tensor,
    visual_sim:  torch.Tensor,
    color_sim:   torch.Tensor | None = None
) -> torch.Tensor:
    """
    Aggregate spatial + visual (+ optional color) similarities.
    Backward-compatible: if color_sim is None, behaves like the original aggregator.
    """
    device = visual_sim.device if torch.is_tensor(visual_sim) else spatial_sim.device
    spatial_sim = spatial_sim.to(device)
    visual_sim  = visual_sim.to(device)
    if color_sim is not None:
        color_sim = color_sim.to(device)

    if str(getattr(cfg, "match_method", "sim_sum")) != "sim_sum":
        raise ValueError(f"Unknown matching method: {cfg.match_method}")

    # keep phys_bias semantics for {spatial, visual}
    pb  = float(cfg.phys_bias)
    w_sp = 1.0 + pb
    w_vs = 1.0 - pb
    w_cl = float(cfg.w_color)
    sims = w_sp * spatial_sim + w_vs * visual_sim
    if color_sim is not None or w_cl == 0.0:
        LOGGER.debug(f"Check color_sim or w_color={w_cl}")
    else:
        sims = sims + w_cl * color_sim
    return sims

@hydra.main(version_base=None, config_path="../configs", config_name="CAPA_slam")
def main(cfg : DictConfig):
    LOGGER.info("START main()")
    _process_cfg(cfg)
    
    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    
    # Load gsa_classes_{}.json file and create gsa_classes_{}_colors.json file
    classes, class_colors = create_or_load_colors(cfg, cfg.color_file_name)

    LOGGER.info(f"Device: {cfg.device} avilable: {torch.cuda.is_available()}")
    objects = MapObjectList(device=cfg.device)
    
    if not cfg.skip_bg:
        # Handle the background detection separately 
        # Each class of them are fused into the map as a single object
        bg_objects = {c: None for c in cfg.bg_classes}
    else:
        bg_objects = None   

    for idx in trange(len(dataset)):
        # Get the color image
        color_path = Path(dataset.color_paths[idx])
        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"
        
        # Get the depth image
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()

        # Get the intrinsics matrix
        cam_K = intrinsics.cpu().numpy()[:3, :3]
        
        # Load the detections
        gobs = None

        if not cfg.part_reg:
            detections_path = color_path.parent.parent / cfg.save_folder_name / 'object' / cfg.detection_folder_name / color_path.name
            detections_path = detections_path.with_suffix(".pkl.gz")
        else:
            detections_path = color_path.parent.parent / cfg.save_folder_name / 'part' / cfg.detection_folder_name / color_path.name
            detections_path = detections_path.with_suffix(".pkl.gz")
        detections_path = str(detections_path)
        
        try:
            with gzip.open(detections_path, "rb") as f:
                gobs = pickle.load(f)
            # LOGGER.debug(f"Loaded detections from {str(color_path.name)}")
        except Exception as e:
            LOGGER.warning(f"Failed to load detections from {str(detections_path)}: {e}")
            continue
        color_path = str(color_path)

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[idx]
        unt_pose = unt_pose.cpu().numpy()
        
        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = cfg,
            image = image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = adjusted_pose,
            class_names = classes,
            BG_CLASSES = cfg.bg_classes,
            color_path = color_path,
            part_reg = cfg.part_reg,
        )
        
        if len(bg_detection_list) > 0:
            for detected_object in bg_detection_list:
                class_name = detected_object['class_name'][0]
                if bg_objects[class_name] is None:
                    bg_objects[class_name] = detected_object
                else:
                    matched_obj = bg_objects[class_name]
                    matched_det = detected_object
                    bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            
        if len(fg_detection_list) == 0:
            continue
            
        if cfg.use_contain_number:
            xyxy = fg_detection_list.get_stacked_values_torch('xyxy', 0).to(cfg.device)
            contain_numbers = compute_2d_box_contained_batch(xyxy, cfg.contain_area_thresh).to(cfg.device)
            for i in range(len(fg_detection_list)):
                fg_detection_list[i]['contain_number'] = [contain_numbers[i]]
            
        if len(objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            continue
        
        if cfg.part_reg and cfg.use_color_feat:
            LOGGER.debug("Using color features for matching")
            spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
            visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
            color_sim = compute_color_similarities(cfg, fg_detection_list, objects)
            agg_sim = aggregate_similarities_wc(cfg, spatial_sim, visual_sim, color_sim)
        else:
            LOGGER.debug("Not using color features for matching")
            spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
            visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
            agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)
        
        # Compute the contain numbers for each detection
        if cfg.use_contain_number:
            # Get the contain numbers for all objects
            contain_numbers_objects = torch.Tensor([obj['contain_number'][0] for obj in objects], device=cfg.device)
            detection_contained = contain_numbers > 0 # (M,)
            object_contained = contain_numbers_objects > 0 # (N,)
            detection_contained = detection_contained.unsqueeze(1) # (M, 1)
            object_contained = object_contained.unsqueeze(0) # (1, N)                

            # Get the non-matching entries, penalize their similarities
            xor = detection_contained ^ object_contained
            agg_sim[xor] = agg_sim[xor] - cfg.contain_mismatch_penalty
        
        # Threshold sims according to cfg. Set to negative infinity if below threshold
        agg_sim[agg_sim < cfg.sim_threshold] = float('-inf')
        
        objects = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim)
        
        # Perform post-processing periodically if told so
        if cfg.denoise_interval > 0 and (idx+1) % cfg.denoise_interval == 0:
            objects = denoise_objects(cfg, objects)
        if cfg.filter_interval > 0 and (idx+1) % cfg.filter_interval == 0:
            objects = filter_objects(cfg, objects)
        if cfg.merge_interval > 0 and (idx+1) % cfg.merge_interval == 0:
            objects = merge_objects(cfg, objects)

    if bg_objects is not None:
        bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
        bg_objects = denoise_objects(cfg, bg_objects)
    
    LOGGER.info("Denoising objects ...")
    objects = denoise_objects(cfg, objects)
    LOGGER.info("Denoising objects ... DONE.")

    # Save the full point cloud before post-processing
    if cfg.save_pcd:
        results = {
            'objects': objects.to_serializable(),
            'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
            'cfg': cfg,
            'class_names': classes,
            'class_colors': class_colors,
        }
        name = f"full_pcd_{cfg.gsa_variant}"
        if cfg.use_color_feat:
            name += "_wc"
        if not cfg.part_reg:
            pcd_save_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / \
                'object' / 'pcd_saves' / f"{name}.pkl.gz"
        else:
            pcd_save_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / \
                'part' / 'pcd_saves' / f"{name}.pkl.gz"
        # make the directory if it doesn't exist
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
        pcd_save_path = str(pcd_save_path)
        
        # with gzip.open(pcd_save_path, "wb") as f:
        #     pickle.dump(results, f)
        # print(f"Saved full point cloud to {pcd_save_path}")
    
    LOGGER.info("Filtering and merging objects ...")
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    LOGGER.info("Filtering and merging objects ... DONE.")
    
    # Save again the full point cloud after the post-processing
    if cfg.save_pcd:
        LOGGER.info("Saving full point cloud ...")
        results['objects'] = objects.to_serializable()
        pcd_save_path = pcd_save_path[:-7] + "_post.pkl.gz"
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        LOGGER.info(f"Saved full point cloud after post-processing to {pcd_save_path}")

    LOGGER.info("FINISH main()")
        
if __name__ == "__main__":
    main()
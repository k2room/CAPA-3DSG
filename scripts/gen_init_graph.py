"""
Generate initial scene graph from the fused 3D objects and parts.
- use CAPA_slam.yaml for configuration (Hydra).
- example:
    $ python scripts/gen_init_graph.py scene_id=0kitchen/video0 dataset=FunGraph3D save_folder_name=capa
"""

import copy, re, json
from collections import defaultdict
import pickle
import gzip
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import trange
import logging

# ===== hydra / omegaconf =====
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

# register custom resolver used in logging filename: ${replace:x,/,-}
OmegaConf.register_new_resolver("replace", lambda s, a, b: str(s).replace(a, b))

LOGGER = logging.getLogger(__name__)  # [HYDRA] use hydra-managed logger

from slam.slam_classes import MapObjectList
from utils.model_utils import (
    aabb_from_obb,
    overlap_1d,
    area_xy,
    overlap_xy_area,
    overlap_xz_area,
    overlap_yz_area,
)

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
    elif ds == "CAPAD":
        cfg.dataset_root   = _resolve_path(cfg.CAPAD_root)
        cfg.dataset_config = _resolve_path(cfg.CAPAD_config)
    else:
        raise ValueError(f"Unknown dataset: {ds}")
    
    OmegaConf.set_struct(cfg, prev_struct)

def _majority_name(names):
    if not names:
        return "unknown"
    arr = np.asarray(names)
    vals, cnts = np.unique(arr, return_counts=True)
    return str(vals[np.argmax(cnts)])

def _ensure_bbox(obj):
    if ('bbox' not in obj or obj['bbox'] is None) and ('pcd' in obj) and (len(obj['pcd'].points) > 0):
        obj['bbox'] = obj['pcd'].get_oriented_bounding_box()

def compute_overlap_ratio(source, target, distance_threshold=0.02):
    # source: part
    # target: object

    # source_tree = o3d.geometry.KDTreeFlann(source)
    target_tree = o3d.geometry.KDTreeFlann(target)
    
    overlap_count = 0
    for point in source.points:
        [_, idx, _] = target_tree.search_radius_vector_3d(point, distance_threshold)
        if len(idx) > 0:
            overlap_count += 1
    
    overlap_ratio = overlap_count / len(source.points)
    return overlap_ratio

@hydra.main(version_base=None, config_path="../configs", config_name="CAPA_slam")
def main(cfg: DictConfig):  
    LOGGER.info("START main()")
    _process_cfg(cfg)
    LOGGER.info(f"Folder: {cfg.save_folder_name} | Generate Initial 3D Scene Graph")

    if cfg.use_color_feat:
        LOGGER.info("Load pkl with color features")
        result_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / 'object' / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_wc_post.pkl.gz"
        part_result_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / 'part' / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_wc_post.pkl.gz"
    else:
        LOGGER.info("Load pkl without color features")
        result_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / 'object' / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_post.pkl.gz"
        part_result_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / 'part' / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_post.pkl.gz"

    # Load fused 3D objects and parts
    # In PKL, 
    #  - Dictionary with keys: ['objects', 'bg_objects', 'cfg', 'class_names', 'class_colors']
    #  - 'objects' is a list of MapObject serializable dictinoaries with keys: 
    #       ['image_idx', 'mask_idx', 'color_path', 'class_name', 'class_id', 'num_detections', 'mask', 'xyxy', 'conf', 'n_points', 
    #        'pixel_area', 'contain_number', 'inst_color', 'is_background', 'clip_ft', 'text_ft', 'color_ft', 'pcd_np', 'bbox_np', 'pcd_color_np']

    LOGGER.info("Load fused 3D objects file")
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    LOGGER.info(f"The number of the fused 3D objects: {len(results['objects'])}")
    
    LOGGER.info("Load fused 3D parts file")
    with gzip.open(part_result_path, "rb") as fp:
        part_results = pickle.load(fp)
    LOGGER.info(f"The number of the fused 3D parts: {len(part_results['objects'])}")
        
    objects = MapObjectList()
    objects.load_serializable(results['objects'])

    parts = MapObjectList()
    parts.load_serializable(part_results['objects'])


    # Vote for refined object/part names: majority class_name
    for obj in objects:
        if ('refined_obj_tag' not in obj) or (not obj['refined_obj_tag']):
            obj['refined_obj_tag'] = _majority_name(obj.get('class_name', []))
        _ensure_bbox(obj)

    for p in parts:
        if ('refined_obj_tag' not in p) or (not p['refined_obj_tag']):
            p['refined_obj_tag'] = _majority_name(p.get('class_name', []))
        _ensure_bbox(p)
    
    # Run the post-processing filtering and merging in instructed to do so
    cfg = copy.deepcopy(results['cfg'])

    # Add parts that can be local connected to objects 
    parts_interest = [
        "knob", "button", "handle", "lid", "dial", "lever", "switch", "faucet", "panel", "hole", "top", "body", "rim", 
        "seat", "armrest", "footrest", "backrest", "screen", "spout", "curtain", "rod", "valve", "head"
        ]

    rigid_inter_id_candidate = []
    part_inter_id_candidate = []

    for inter_idx, obj_inter in enumerate(objects):
        obj_inter['connected_parts'] = []
    
    # Iterate over all objects and parts to find connections
    for inter_idx in trange(len(objects)):
        obj_inter = objects[inter_idx]
        obj_classes_inter = np.asarray(obj_inter['class_name'])
        values_inter, counts_inter = np.unique(obj_classes_inter, return_counts=True)
        obj_class_inter = values_inter[np.argmax(counts_inter)]
        tag = False
        for part_idx, part in enumerate(parts):
            part_classes = np.asarray(part['class_name'])
            values, counts = np.unique(part_classes, return_counts=True)
            part_class = values[np.argmax(counts)]
            part_class = re.findall(r"[a-z]+", str(part_class).lower())
            for p in parts_interest:
                if p in part_class:
                    points_part = part['pcd']
                    points_obj_inter = obj_inter['pcd']
                    iou = compute_overlap_ratio(points_part, points_obj_inter, 0.02)
                    # fusion based on inter objects: 1 object many parts and object must be big enough
                    obj_box_extent = obj_inter['bbox'].extent
                    part_box_extent = part['bbox'].extent
                    if (iou > 0.7 and obj_box_extent.mean() > 2 * part_box_extent.mean()) or (iou > 0.1 and obj_class_inter in part_class):
                        LOGGER.debug(f"{obj_class_inter}, {part_class} {iou}")
                        if 'connected_parts' not in obj_inter:
                            obj_inter['connected_parts'] = []
                            obj_inter['connected_parts'].append(part_idx)
                            part_inter_id_candidate.append(part_idx)
                            tag = True
                        else:
                            obj_inter['connected_parts'].append(part_idx)
                            part_inter_id_candidate.append(part_idx)
                            tag = True
        if tag:
            rigid_inter_id_candidate.append(inter_idx)

    LOGGER.info("Generating initial 3D scene graph...")
    # Generate Initial 3D Scene Graph
    obj_map, part_map = {}, {}

    # Object/Part Node
    for i, obj in enumerate(objects):
        obb = obj['bbox']
        c = [float(x) for x in np.round(np.asarray(obb.center), 3)]
        e = [float(x) for x in np.round(np.asarray(obb.extent), 3)]
        oid = f"obj_{i}"
        label = str(obj['refined_obj_tag'])
        nd = {"label": label, "center": c, "extent": e}
        if 'connected_parts' in obj and obj['connected_parts']:
            nd["connected_parts"] = [f"part_{int(k)}" for k in obj['connected_parts']]
        obj_map[oid] = nd

    for j, p in enumerate(parts):
        obb = p['bbox']
        c = [float(x) for x in np.round(np.asarray(obb.center), 3)]
        e = [float(x) for x in np.round(np.asarray(obb.extent), 3)]
        pid = f"part_{j}"
        label = str(p['refined_obj_tag'])
        part_map[pid] = {"label": label, "center": c, "extent": e}

    # Spatial Relation (subject → [[object, relation], ...])
    TH_AXIS = 1.0
    TOUCH  = 0.05   # 5cm
    spatial = defaultdict(list)

    # AABB cache for checking relations
    aabbs = {}
    for i in range(len(objects)):
        aabbs[f"obj_{i}"] = aabb_from_obb(objects[i]['bbox'])
    for j in range(len(parts)):
        aabbs[f"part_{j}"] = aabb_from_obb(parts[j]['bbox'])

    def _add_rel(obj_id, rel, subj_id):
        # avoid duplicates
        pair = [obj_id, rel]
        lst = spatial[subj_id]
        if pair not in lst:
            lst.append(pair)

    # check all object pairs
    # nodes_all = list(obj_map.keys()) + list(part_map.keys())
    nodes_all = list(obj_map.keys())
    for A in nodes_all:
        for B in nodes_all:
            if A == B:
                continue
            mnA, mxA = aabbs[A]
            mnB, mxB = aabbs[B]

            # above / below (XY overlap + z-axis gap)
            xy_area = overlap_xy_area(mnA, mxA, mnB, mxB)
            area_min = max(1e-9, min(area_xy(mnA, mxA), area_xy(mnB, mxB)))
            overlap_ratio = xy_area / area_min
            sep_z_above = mnA[2] - mxB[2]   # A is above B
            sep_z_below = mnB[2] - mxA[2]   # A is under B

            if xy_area > 0 and 0.0 <= sep_z_above <= TH_AXIS:
                _add_rel(A, "above", B)
                if sep_z_above <= TOUCH:
                    if overlap_ratio >= 0.8:
                        _add_rel(A, "cover", B)
                    elif overlap_ratio >= 0.6:
                        _add_rel(A, "lying_on", B)
                    elif overlap_ratio <= 0.2:
                        _add_rel(A, "standing_on", B)
                    if mnA[2] < mxB[2]:
                        _add_rel(A, "hanging_on", B)
                else:
                    _add_rel(A, "on_top_of", B)

            if xy_area > 0 and 0.0 <= sep_z_below <= TH_AXIS:
                _add_rel(A, "below", B)

            # front / behind (XZ overlap + y-axis gap)
            xz_area = overlap_xz_area(mnA, mxA, mnB, mxB)
            sep_y_front  = mnA[1] - mxB[1]
            sep_y_behind = mnB[1] - mxA[1]
            if xz_area > 0 and 0.0 <= sep_y_front <= TH_AXIS:
                _add_rel(A, "front", B)
            if xz_area > 0 and 0.0 <= sep_y_behind <= TH_AXIS:
                _add_rel(A, "behind", B)

            # right / left (YZ overlap + x-axis gap)
            yz_area = overlap_yz_area(mnA, mxA, mnB, mxB)
            sep_x_right = mnA[0] - mxB[0]
            sep_x_left  = mnB[0] - mxA[0]
            if yz_area > 0 and 0.0 <= sep_x_right <= TH_AXIS:
                _add_rel(A, "right", B)
            if yz_area > 0 and 0.0 <= sep_x_left <= TH_AXIS:
                _add_rel(A, "left", B)

            # nearby (center distance)
            cA = np.asarray(objects[int(A.split('_')[1])]['bbox'].center) if A.startswith("obj_") \
                else np.asarray(parts[int(A.split('_')[1])]['bbox'].center)
            cB = np.asarray(objects[int(B.split('_')[1])]['bbox'].center) if B.startswith("obj_") \
                else np.asarray(parts[int(B.split('_')[1])]['bbox'].center)
            if np.linalg.norm(cA - cB) <= TH_AXIS:
                _add_rel(A, "nearby", B)

    # JSON format output
    scene_graph = {
        "object": obj_map,
        "part": part_map,
        "spatial_relation": dict(spatial),
        "functional_relation": []
    }
    sg_dir = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / "scene_graph"
    sg_dir.mkdir(parents=True, exist_ok=True)
    sg_path = sg_dir / "initial_3d_scene_graph.json"
    with open(sg_path, "w") as jf:
        json.dump(scene_graph, jf, indent=2)
    LOGGER.info(f"[SceneGraph] Saved: {str(sg_path)}")


    # Save updated results
    updated_results = {
        'objects': objects.to_serializable(),
        'cfg': results['cfg'],
        'class_names': results['class_names'],
        'class_colors': results['class_colors'],
        'inter_id_candidate': rigid_inter_id_candidate
    }    
    
    LOGGER.info("Saving updated fused 3D objects file")
    save_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / \
        'object' / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_update.pkl.gz"
    
    with gzip.open(str(save_path), "wb") as f:
        pickle.dump(updated_results, f)
    LOGGER.info(f"Saved full point cloud to {str(save_path)}")

    updated_results = {
        'objects': parts.to_serializable(),
        'cfg': part_results['cfg'],
        'class_names': part_results['class_names'],
        'class_colors': part_results['class_colors'],
        'part_inter_id_candidate': part_inter_id_candidate
    }    
    
    LOGGER.info("Saving updated fused 3D parts file")
    save_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / \
        'part' / 'pcd_saves' /  f"full_pcd_{cfg.gsa_variant}_update.pkl.gz"
    
    with gzip.open(str(save_path), "wb") as f:
        pickle.dump(updated_results, f)
    LOGGER.info(f"Saved full point cloud to {str(save_path)}")       

    LOGGER.info("FINISH main()")

if __name__ == "__main__":
    main()
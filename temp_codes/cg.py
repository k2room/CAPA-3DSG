from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox, 
    orr_log_rgb_image, 
    orr_log_vlm_image
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_obj_rel_from_image_gpt4v, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, 
    find_existing_image_path, 
    get_det_out_path, 
    get_exp_out_path, 
    get_vlm_annotated_image_path, 
    handle_rerun_saving, 
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges_and_captions, 
    measure_time, 
    save_detection_results,
    save_edge_json, 
    save_hydra_config,
    save_obj_json, 
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs as _base_make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    # merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs, 
    filter_captions,
    tracker as _mapping_tracker,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    # merge_obj_matches
)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections

import torch
import torch.nn.functional as F
import numpy as np
import logging
import uuid
from typing import List, Optional

from conceptgraph.slam.slam_classes import DetectionList, to_tensor

owandb = OptionalWandB()
tracker = MappingTracker()

def filter_gobs(
    gobs: dict,
    image: np.ndarray,
    skip_bg: bool = None,  # Explicitly passing skip_bg
    BG_CLASSES: list = None,  # Explicitly passing BG_CLASSES
    mask_area_threshold: float = 10,  # Default value as fallback
    max_bbox_area_ratio: float = None,  # Explicitly passing max_bbox_area_ratio
    mask_conf_threshold: float = None,  # Explicitly passing mask_conf_threshold
    return_index_map: bool = False,     
):
    # # If no detection at all
    # if len(gobs['xyxy']) == 0:
    #     return gobs
    if len(gobs['xyxy']) == 0:
        return (gobs, {}) if return_index_map else gobs

    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]

        # Skip masks that are too small
        mask_area = gobs['mask'][mask_idx].sum()
        if mask_area < max(mask_area_threshold, 10):
            logging.debug(f"Skipped due to small mask area ({mask_area} pixels) - Class: {class_name}")
            continue

        # Skip the BG classes
        if skip_bg and class_name in BG_CLASSES:
            logging.debug(f"Skipped background class: {class_name}")
            continue

        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if max_bbox_area_ratio is not None and bbox_area > max_bbox_area_ratio * image_area:
                logging.debug(f"Skipped due to large bounding box area ratio - Class: {class_name}, Area Ratio: {bbox_area/image_area:.4f}")
                continue

        # Skip masks with low confidence
        if mask_conf_threshold is not None and gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < mask_conf_threshold:
                # logging.debug(f"Skipped due to low confidence ({gobs['confidence'][mask_idx]}) - Class: {class_name}")
                continue

        idx_to_keep.append(mask_idx)

    # for key in gobs.keys():
    #     print(key, type(gobs[key]), len(gobs[key]))

    # for attribute in gobs.keys():
    for attribute in list(gobs.keys()):  # avoid dict-size-change during iteration
        if isinstance(gobs[attribute], str) or attribute == "classes":  # Captions
            continue
        if attribute in ['labels', 'edges', 'text_feats', 'captions']:
            # Note: this statement was used to also exempt 'detection_class_labels' but that causes a bug. It causes the edges to be misalgined with the objects.
            continue
        elif isinstance(gobs[attribute], list):
            gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
        elif isinstance(gobs[attribute], np.ndarray):
            gobs[attribute] = gobs[attribute][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")
        
    # filtered_captions = filter_captions(gobs['captions'], gobs['detection_class_labels'])
    # gobs['captions'] = filtered_captions

    # return gobs
    filtered_captions = filter_captions(gobs['captions'], gobs['detection_class_labels'])
    gobs['captions'] = filtered_captions

    if return_index_map:
        idx_map = {old_i: new_i for new_i, old_i in enumerate(idx_to_keep)}
        return gobs, idx_map
    else:
        return gobs

# @profile
def merge_obj2_into_obj1(obj1, obj2, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type, device, run_dbscan=True):

    '''
    Merges obj2 into obj1 with structured attribute handling, including explicit checks for unhandled keys.

    Parameters:
    - obj1, obj2: Objects to merge.
    - downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type: Parameters for point cloud processing.
    - device: Computation device.
    - run_dbscan: Whether to run DBSCAN for noise removal.

    Returns:
    - obj1: Updated object after merging.
    '''
    global tracker
    
    tracker.track_merge(obj1, obj2)
    
    # Attributes to be explicitly handled
    extend_attributes = ['image_idx', 'mask_idx', 'color_path', 'class_id', 'mask', 'xyxy', 'conf', 'contain_number', 'captions']
    add_attributes = ['num_detections', 'num_obj_in_class']
    skip_attributes = ['id', 'class_name', 'is_background', 'new_counter', 'curr_obj_num', 'inst_color', 'is_part', 'det_type']  # 'inst_color' just keeps obj1's
    custom_handled = ['pcd', 'bbox', 'clip_ft', 'text_ft', 'n_points']

    # Check for unhandled keys and throw an error if there are
    all_handled_keys = set(extend_attributes + add_attributes + skip_attributes + custom_handled)
    unhandled_keys = set(obj2.keys()) - all_handled_keys
    if unhandled_keys:
        raise ValueError(f"Unhandled keys detected in obj2: {unhandled_keys}. Please update the merge function to handle these attributes.")

    # Custom handling for 'pcd', 'bbox', 'clip_ft', and 'text_ft'
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    # Process extend and add attributes
    for attr in extend_attributes:
        if attr in obj1 and attr in obj2:
            obj1[attr].extend(obj2[attr])
    
    for attr in add_attributes:
        if attr in obj1 and attr in obj2:
            obj1[attr] += obj2[attr]

    # Handling 'caption'
    if 'caption' in obj1 and 'caption' in obj2:
        # n_obj1_det = obj1['num_detections']
        for key, value in obj2['caption'].items():
            obj1['caption'][key + n_obj1_det] = value

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan)
    # update n_points
    obj1['n_points'] = len(np.asarray(obj1['pcd'].points))

    # Update 'bbox'
    obj1['bbox'] = get_bounding_box(spatial_sim_type, obj1['pcd'])
    obj1['bbox'].color = [0, 1, 0]

    # Merge and normalize 'clip_ft'
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det + obj2['clip_ft'] * n_obj2_det) / (n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    return obj1

def make_detection_list_from_pcd_and_gobs(
    obj_pcds_and_bboxes,
    gobs: dict,
    color_path,
    obj_classes,
    image_idx,
    idx_obj_filt=None,
    idx_part_filt=None,
):
    """
    CAPA-aware override.
    Builds DetectionList with per-detection type flags (object vs part) while
    preserving ConceptGraph's expected fields.

    Compatible with original signature; extra indices are optional.
    """
    # Fast path: if no special handling requested, delegate to base
    if idx_obj_filt is None and idx_part_filt is None:
        return _base_make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, image_idx
        )

    det_list = DetectionList()

    # Build quick lookup for part/object indices
    obj_set = set(int(i) for i in (idx_obj_filt or []))
    part_set = set(int(i) for i in (idx_part_filt or []))

    # For counters/metadata compatibility with upstream
    tracker = _mapping_tracker

    for mask_idx in range(len(gobs.get('mask', []))):
        entry = obj_pcds_and_bboxes[mask_idx] if mask_idx < len(obj_pcds_and_bboxes) else None
        if entry is None:
            continue

        try:
            curr_class_name = gobs['classes'][int(gobs['class_id'][mask_idx])]
        except Exception:
            curr_class_name = str(gobs['class_id'][mask_idx]) if 'class_id' in gobs else 'obj'
        try:
            curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)
        except ValueError:
            curr_class_idx = -1

        is_bg_object = bool(curr_class_name in obj_classes.get_bg_classes_arr())
        num_obj_in_class = tracker.curr_class_count.get(curr_class_name, 0)

        det = {
            'id': uuid.uuid4(),
            'image_idx': [image_idx],
            'mask_idx': [mask_idx],
            'color_path': [color_path],
            'class_name': curr_class_name,
            'class_id': [curr_class_idx],
            'captions': [gobs.get('captions', [None])[mask_idx] if 'captions' in gobs else None],
            'num_detections': 1,
            'mask': [gobs['mask'][mask_idx]],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs.get('confidence', [None])[mask_idx] if 'confidence' in gobs else None],
            'n_points': len(entry['pcd'].points),
            'contain_number': [None],
            'inst_color': np.random.rand(3),
            'is_background': is_bg_object,

            'pcd': entry['pcd'],
            'bbox': entry['bbox'],
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]) if 'image_feats' in gobs else None,
            'num_obj_in_class': num_obj_in_class,
            'curr_obj_num': tracker.total_object_count,
            'new_counter': tracker.brand_new_counter,
        }

        # Tag detection type for CAPA weighting
        if mask_idx in part_set:
            det['det_type'] = 'part'
            det['is_part'] = True
        elif mask_idx in obj_set:
            det['det_type'] = 'object'
            det['is_part'] = False
        else:
            det['det_type'] = 'object'
            det['is_part'] = False

        det_list.append(det)

        # keep upstream counters consistent
        tracker.curr_class_count[curr_class_name] = tracker.curr_class_count.get(curr_class_name, 0) + 1
        tracker.total_object_count += 1
        tracker.brand_new_counter += 1

    return det_list

# @profile
def merge_overlap_objects(
    merge_overlap_thresh: float,
    merge_visual_sim_thresh: float,
    merge_text_sim_thresh: float,
    objects: MapObjectList,
    overlap_matrix: np.ndarray,
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    map_edges = None,
):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]
    
    # Sort indices of overlap ratios in descending order
    sort = np.argsort(overlap_ratio)[::-1]  
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]
    
    merge_operations = []  # to track merge operations
    kept_objects = np.ones(
        len(objects), dtype=bool
    )  # Initialize all objects as 'kept' initially
    
    index_updates = list(range(len(objects)))  # Initialize index updates with the same indices

    for i, j, ratio in zip(x, y, overlap_ratio):
        if ratio > merge_overlap_thresh:
            visual_sim = F.cosine_similarity(
                to_tensor(objects[i]["clip_ft"]),
                to_tensor(objects[j]["clip_ft"]),
                dim=0,
            )
            # text_sim = F.cosine_similarity(
            #     to_tensor(objects[i]["text_ft"]),
            #     to_tensor(objects[j]["text_ft"]),
            #     dim=0,
            # )
            text_sim = visual_sim
            if (visual_sim > merge_visual_sim_thresh) and (text_sim > merge_text_sim_thresh):
                if kept_objects[j]:  # Check if the target object has not been merged into another
                    # Merge object i into object j
                    objects[j] = merge_obj2_into_obj1(
                        objects[j],
                        objects[i],
                        downsample_voxel_size,
                        dbscan_remove_noise,
                        dbscan_eps,
                        dbscan_min_points,
                        spatial_sim_type,
                        device,
                        run_dbscan=True,
                    )
                    kept_objects[i] = False  # Mark object i as 'merged'
                    merge_operations.append((i, j))  # Record this merge for edge updates 
                    index_updates[i] = None  # Update index as merged
        else:
            break  # Stop processing if the current overlap ratio is below the threshold
        
    # Update remaining indices in index_updates
    current_index = 0
    for original_index, is_kept in enumerate(kept_objects):
        if is_kept:
            index_updates[original_index] = current_index
            current_index += 1
        else:
            index_updates[original_index] = None

    # Create a new list of objects excluding those that were merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)

    return objects, index_updates

# @profile
def merge_objects(
    merge_overlap_thresh: float,
    merge_visual_sim_thresh: float,
    merge_text_sim_thresh: float,
    objects: MapObjectList,
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    do_edges: bool = False,
    map_edges = None,
):
    if len(objects) == 0:
        return objects
    if merge_overlap_thresh <= 0:
        return objects

    # Assuming compute_overlap_matrix requires only `objects` and `downsample_voxel_size`
    overlap_matrix = compute_overlap_matrix_general(
        objects_a=objects,
        objects_b=None,
        downsample_voxel_size=downsample_voxel_size,
    )
    print("Before merging:", len(objects))

    objects, index_updates = merge_overlap_objects(
        merge_overlap_thresh=merge_overlap_thresh,
        merge_visual_sim_thresh=merge_visual_sim_thresh,
        merge_text_sim_thresh=merge_text_sim_thresh,
        objects=objects,
        overlap_matrix=overlap_matrix,
        downsample_voxel_size=downsample_voxel_size,
        dbscan_remove_noise=dbscan_remove_noise,
        dbscan_eps=dbscan_eps,
        dbscan_min_points=dbscan_min_points,
        spatial_sim_type=spatial_sim_type,
        device=device,
        map_edges=map_edges,
    )
    

    if do_edges:
        map_edges.merge_update_indices(index_updates)
        map_edges.update_objects_list(objects)
        print("After merging:", len(objects))


    if do_edges:
        return objects, map_edges
    else:
        return objects

def merge_obj_matches(
    detection_list: DetectionList,
    objects: MapObjectList,
    match_indices: List[Optional[int]],
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
) -> MapObjectList:
    """
    Merges detected objects into existing objects based on a list of match indices.

    Args:
        detection_list (DetectionList): List of detected objects.
        objects (MapObjectList): List of existing objects.
        match_indices (List[Optional[int]]): Indices of existing objects each detected object matches with.
        downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type, device:
            Parameters for merging and similarity computation.

    Returns:
        MapObjectList: Updated list of existing objects with detected objects merged as appropriate.
    """
    global tracker
    temp_curr_object_count = tracker.curr_object_count
    for detected_obj_idx, existing_obj_match_idx in enumerate(match_indices):
        if existing_obj_match_idx is None:
            # track the new object detection
            tracker.object_dict.update({
                "id": detection_list[detected_obj_idx]['id'],
                "first_discovered": tracker.curr_frame_idx
            })

            objects.append(detection_list[detected_obj_idx])
        else:

            detected_obj = detection_list[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            merged_obj = merge_obj2_into_obj1(
                obj1=matched_obj,
                obj2=detected_obj,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=dbscan_remove_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
                spatial_sim_type=spatial_sim_type,
                device=device,
                run_dbscan=False,
            )
            objects[existing_obj_match_idx] = merged_obj
    tracker.increment_total_merges(len(match_indices) - match_indices.count(None))
    tracker.increment_total_objects(len(objects) - temp_curr_object_count)
    # wandb.log({"merges_this_frame" :len(match_indices) - match_indices.count(None)})
    # wandb.log({"total_merges": tracker.total_merges})
    owandb.log(
        {
            "merges_this_frame": len(match_indices) - match_indices.count(None),
            "total_merges": tracker.total_merges,
            "frame_idx": tracker.curr_frame_idx,
        }
    )
    return objects

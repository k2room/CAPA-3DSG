import json
import open3d as o3d
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse
import pickle
import gzip
from openfungraph.slam.slam_classes import MapObjectList
from tqdm import tqdm
from openfungraph.utils.ious import compute_3d_iou


CLASS_LABELS_FUNC = ["button / knob",  "power strip", "light switch", "foucet / handle", "button", "handle", "knob", "knob / button", "foucet / knob / handle", "switch panel / electric outlet", "remote", "electric outlet / power strip", "handle / foucet", "switch panel", "electric outlet"]


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--iou_threshold", type=float, default=0.)

    return parser


def load_result(result_path):

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    
    objects = MapObjectList()
    objects.load_serializable(results["objects"])

    if 'bg_objects' not in results:
        bg_objects = None
    elif results['bg_objects'] is None:
        bg_objects = None
    else:
        bg_objects = MapObjectList()
        bg_objects.load_serializable(results["bg_objects"])

    class_colors = results['class_colors']
    class_names = results['class_names']
    try:
        obj_cand = results['inter_id_candidate']
    except:
        obj_cand = results['part_inter_id_candidate']

    return objects, bg_objects, class_colors, class_names, obj_cand


def compute_iou(pc1, pc2, voxel_size):
    # Voxelization
    pcd1 = pc1.voxel_down_sample(voxel_size)
    pcd2 = pc2.voxel_down_sample(voxel_size)

    # Create binary occupancy grids
    min_bound = np.minimum(pcd1.get_min_bound(), pcd2.get_min_bound())
    max_bound = np.maximum(pcd1.get_max_bound(), pcd2.get_max_bound())
    
    # Define grid size
    grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    # Create occupancy grid
    grid1 = np.zeros(grid_size, dtype=bool)
    grid2 = np.zeros(grid_size, dtype=bool)

    # Fill occupancy grids
    for point in np.asarray(pcd1.points):
        grid_idx = np.floor((point - min_bound) / voxel_size).astype(int)
        grid1[tuple(grid_idx)] = True

    for point in np.asarray(pcd2.points):
        grid_idx = np.floor((point - min_bound) / voxel_size).astype(int)
        grid2[tuple(grid_idx)] = True

    # Calculate intersection and union
    intersection = np.sum(grid1 & grid2)
    union = np.sum(grid1 | grid2)

    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.dataset == 'SceneFun3D':
        with open(args.root_path+'/SceneFun3D.annotations.json', 'r') as f:
            gt_annos = json.load(f)
        gt_annos = [anno for anno in gt_annos if anno['scene_id'] == args.scene]
    elif args.dataset == 'FunGraph3D':
        with open(args.root_path+'/FunGraph3D.annotations.json', 'r') as f:
            gt_annos = json.load(f)
        gt_annos = [anno for anno in gt_annos if anno['scene_id'] == args.scene]
    else:
        exit(1)
    
    if args.dataset == 'SceneFun3D':
        scene_pc = o3d.io.read_point_cloud(args.root_path+'/scans/'+args.scene+'_laser_scan.ply')
        refined_transform = np.load(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/'+args.video+'_refined_transform.npy') 
        scene_pc.transform(refined_transform)
    elif args.dataset == 'FunGraph3D':
        scene_pc = o3d.io.read_point_cloud(args.root_path+'/scans/'+args.scene+'.ply')
    
    if args.dataset == 'SceneFun3D':
        objects, _, _, _, _ = load_result(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz')
        parts, _, _, _, _ = load_result(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/part/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz')
        with open(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/cfslam_funcgraph_edges.pkl', "rb") as f:
            edges = pickle.load(f)
    elif args.dataset == 'FunGraph3D':
        objects, _, _, _, _ = load_result(args.root_path+'/'+args.scene+'/'+args.video+'/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz')
        parts, _, _, _, _ = load_result(args.root_path+'/'+args.scene+'/'+args.video+'/part/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz')
        with open(args.root_path+'/'+args.scene+'/'+args.video+'/cfslam_funcgraph_edges.pkl', "rb") as f:
            edges = pickle.load(f)

    all_labels_embeddings = np.load(args.root_path+'/all_labels_clip_embeddings.npy')
    with open(args.root_path+'/all_labels.json', 'r') as f:
        all_labels = json.load(f)
    
    obj_idx = []
    part_idx = []
    for edge in edges:
        if edge[2] == -1:
            # obj + part
            obj_idx.append(edge[0])
            part_idx.append(edge[1])
        elif edge[1] == -1:
            # obj + obj
            obj_idx.append(edge[0])
            obj_idx.append(edge[2])
    obj_idx = list(set(obj_idx))
    part_idx = list(set(part_idx))

    rk_num_obj = 0
    rk_num_func = 0
    r10_num_obj = 0
    r10_num_func = 0
    total_obj_num = 0
    total_func_num = 0

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    for gt in tqdm(gt_annos):
        gt_label = gt['label']
        if gt_label in CLASS_LABELS_FUNC:
            total_func_num += 1
        else:
            total_obj_num += 1
        gt_mask = gt['indices']
        gt_pc = np.asarray(scene_pc.points)[gt_mask]
        gt_pc_o3d = o3d.geometry.PointCloud()
        gt_pc_o3d.points = o3d.utility.Vector3dVector(gt_pc)
        gt_bbd = gt_pc_o3d.get_oriented_bounding_box()
        corr_flag = False
        for obj_id in obj_idx:
            pred_bbd = objects[obj_id]['bbox']
            pred_label = objects[obj_id]['refined_obj_tag']
            iou = compute_3d_iou(gt_bbd, pred_bbd)
            if iou > args.iou_threshold:
                inputs = processor(text=[pred_label], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model.get_text_features(**inputs)
                embeddings = embeddings.detach().numpy()
                norm_embeddings = embeddings / np.linalg.norm(embeddings)
                norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
                topk_indices = np.argsort(similarity[0], axis=0)[-args.topk:][::-1]
                topk_label = [all_labels[idx] for idx in topk_indices]
                if gt_label in topk_label and gt_label in CLASS_LABELS_FUNC:
                    rk_num_func += 1
                    corr_flag = True
                    break
                elif gt_label in topk_label:
                    rk_num_obj += 1
                    corr_flag = True
                    break

        if corr_flag:
            continue
        for part_id in part_idx:
            pred_bbd = parts[part_id]['bbox']
            pred_label = parts[part_id]['refined_obj_tag']
            iou = compute_3d_iou(gt_bbd, pred_bbd)
            if iou > args.iou_threshold:
                inputs = processor(text=[pred_label], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model.get_text_features(**inputs)
                embeddings = embeddings.detach().numpy()
                norm_embeddings = embeddings / np.linalg.norm(embeddings)
                norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
                topk_indices = np.argsort(similarity[0], axis=0)[-args.topk:][::-1]
                topk_label = [all_labels[idx] for idx in topk_indices]
                if gt_label in topk_label and gt_label in CLASS_LABELS_FUNC:
                    rk_num_func += 1
                    corr_flag = True
                    break
                elif gt_label in topk_label:
                    rk_num_obj += 1
                    corr_flag = True
                    break
        if corr_flag:
            continue
    
    for gt in tqdm(gt_annos):
        gt_label = gt['label']
        gt_mask = gt['indices']
        gt_pc = np.asarray(scene_pc.points)[gt_mask]
        gt_pc_o3d = o3d.geometry.PointCloud()
        gt_pc_o3d.points = o3d.utility.Vector3dVector(gt_pc)
        gt_bbd = gt_pc_o3d.get_oriented_bounding_box()
        corr_flag = False
        for obj_id in obj_idx:
            pred_bbd = objects[obj_id]['bbox']
            pred_label = objects[obj_id]['refined_obj_tag']
            iou = compute_3d_iou(gt_bbd, pred_bbd)
            if iou > args.iou_threshold:
                inputs = processor(text=[pred_label], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model.get_text_features(**inputs)
                embeddings = embeddings.detach().numpy()
                norm_embeddings = embeddings / np.linalg.norm(embeddings)
                norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
                topk_indices = np.argsort(similarity[0], axis=0)[-10:][::-1]
                topk_label = [all_labels[idx] for idx in topk_indices]
                if gt_label in topk_label and gt_label in CLASS_LABELS_FUNC:
                    r10_num_func += 1
                    corr_flag = True
                    break
                elif gt_label in topk_label:
                    r10_num_obj += 1
                    corr_flag = True
                    break

        if corr_flag:
            continue
        for part_id in part_idx:
            pred_bbd = parts[part_id]['bbox']
            pred_label = parts[part_id]['refined_obj_tag']
            iou = compute_3d_iou(gt_bbd, pred_bbd)
            if iou > args.iou_threshold:
                inputs = processor(text=[pred_label], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model.get_text_features(**inputs)
                embeddings = embeddings.detach().numpy()
                norm_embeddings = embeddings / np.linalg.norm(embeddings)
                norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
                topk_indices = np.argsort(similarity[0], axis=0)[-10:][::-1]
                topk_label = [all_labels[idx] for idx in topk_indices]
                if gt_label in topk_label and gt_label in CLASS_LABELS_FUNC:
                    r10_num_func += 1
                    corr_flag = True
                    break
                elif gt_label in topk_label:
                    r10_num_obj += 1
                    corr_flag = True
                    break
        if corr_flag:
            continue
    
    print('Top ', args.topk, ' Obj Recall: ', rk_num_obj , ' / ' , total_obj_num , ': ' , rk_num_obj / total_obj_num)
    print('Top ', args.topk, ' Fun Elements Recall: ', rk_num_func , ' / ' , total_func_num , ': ' , rk_num_func / total_func_num)
    print('Top ', args.topk, ' Overall Recall: ', (rk_num_func + rk_num_obj) / (total_obj_num + total_func_num))
    print('Top 10 Obj Recall: ', r10_num_obj , ' / ' , total_obj_num , ': ' , r10_num_obj / total_obj_num)
    print('Top 10 Fun Elements Recall: ', r10_num_func , ' / ' , total_func_num , ': ' , r10_num_func / total_func_num)
    print('Top 10 Overall Recall: ', (r10_num_func + r10_num_obj) / (total_obj_num + total_func_num))
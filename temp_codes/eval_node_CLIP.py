"""
    conda install -c conda-forge sentence-transformers==5.1.2
    pip install "transformers==4.35.2"
    pip install omegaconf open3d

    Usage Example:
        python eval/eval_all_SBERT.py --dataset FunGraph3D --root_path /home/main/workspace/k2room2/gpuserver00_storage/CAPA/FunGraph3D --scene 6kitchen --video video1 --gpt gpt40613
        python eval/eval_all_SBERT.py --dataset SceneFun3D --root_path /home/main/workspace/k2room2/gpuserver00_storage/CAPA/SceneFun3D_Graph --scene 422391 --video 42446522 --split dev --gpt gpt520250807
"""
import json
import open3d as o3d
import torch
import torch.nn.functional as F
from collections.abc import Iterable
import copy
from transformers import CLIPProcessor, CLIPModel
from transformers import BertModel, BertTokenizer
import numpy as np
import argparse
import pickle
import gzip
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

CLASS_LABELS_FUNC = ["button / knob",  "power strip", "light switch", "foucet / handle", "button", "handle", "knob", "knob / button", "foucet / knob / handle", "switch panel / electric outlet", "remote", "electric outlet / power strip", "handle / foucet", "switch panel", "electric outlet"]

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)

class DetectionList(list):
    def get_values(self, key, idx:int=None):
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]
    
    def get_stacked_values_torch(self, key, idx:int=None):
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)
    
    def get_stacked_values_numpy(self, key, idx:int=None):
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)
    
    def __add__(self, other):
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list
    
    def __iadd__(self, other):
        self.extend(other)
        return self
    
    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist of the current list by indexing
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self
    
    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist of the current list by masking
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self
    
    def get_most_common_class(self) -> list[int]:
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_name']), return_counts=True)
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes
    
    def color_by_most_common_classes(self, colors_dict: dict[str, list[float]], color_bbox: bool=True):
        '''
        Color the point cloud of each detection by the most common class
        '''
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            color = colors_dict[str(c)]
            d['pcd'].paint_uniform_color(color)
            if color_bbox:
                d['bbox'].color = color
                
    def color_by_instance(self):
        if len(self) == 0:
            # Do nothing
            return
        
        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]

class MapObjectList(DetectionList):
    def compute_similarities(self, new_clip_ft):
        '''
        The input feature should be of shape (D, ), a one-row vector
        This is mostly for backward compatibility
        '''
        # if it is a numpy array, make it a tensor 
        new_clip_ft = to_tensor(new_clip_ft)
        
        # assuming cosine similarity for features
        clip_fts = self.get_stacked_values_torch('clip_ft')

        similarities = F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)
        # return similarities.squeeze()
        return similarities
    
    def to_serializable(self):
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)
            
            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])
            
            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            
            del s_obj_dict['pcd']
            del s_obj_dict['bbox']
            
            s_obj_list.append(s_obj_dict)
            
        return s_obj_list
    
    def load_serializable(self, s_obj_list):
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            try:
                new_obj = copy.deepcopy(s_obj_dict)
                
                new_obj['clip_ft'] = to_tensor(new_obj['clip_ft'])
                new_obj['text_ft'] = to_tensor(new_obj['text_ft'])
                
                new_obj['pcd'] = o3d.geometry.PointCloud()
                new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
                new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(new_obj['bbox_np']))
                new_obj['bbox'].color = new_obj['pcd_color_np'][0]
                new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])
                
                del new_obj['pcd_np']
                del new_obj['bbox_np']
                del new_obj['pcd_color_np']
                
                self.append(new_obj)
            except:
                continue

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--iou_threshold", type=float, default=0.)
    parser.add_argument("--gpt", type=str, default='gpt40613')       # Folder name: gpt40613, gpt520250807, ... located in each scene folder
    parser.add_argument("--debug", type=bool, default=False)

    parser.add_argument("--edge_file", type=str, default='')
    parser.add_argument("--obj_file", type=str, default='')
    parser.add_argument("--part_file", type=str, default='')

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

def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    
    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    if use_iou:
        return iou
    else:
        return max_overlap


def _majority_name(names):
    if not names:
        return "unknown"
    arr = np.asarray(names)
    vals, cnts = np.unique(arr, return_counts=True)
    return str(vals[np.argmax(cnts)])

def _majority_id(ids):
    if not ids:
        return -1
    arr = np.asarray(ids)
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(cnts)])

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.dataset == 'SceneFun3D':
        with open(args.root_path+'/SceneFun3D.annotations.json', 'r') as f:
            gt_annos = json.load(f)
        gt_annos = [anno for anno in gt_annos if anno['scene_id'] == args.scene]
        with open(args.root_path+'/SceneFun3D.relations.refined.json', 'r') as f:
            gt_edge_annos = json.load(f)
        gt_edge_annos = [anno for anno in gt_edge_annos if anno['scene_id'] == args.scene]
    elif args.dataset == 'FunGraph3D':
        with open(args.root_path+'/FunGraph3D.annotations.json', 'r') as f:
            gt_annos = json.load(f)
        gt_annos = [anno for anno in gt_annos if anno['scene_id'] == args.scene]
        with open(args.root_path+'/FunGraph3D.relations.refined.json', 'r') as f:
            gt_edge_annos = json.load(f)
        gt_edge_annos = [anno for anno in gt_edge_annos if anno['scene_id'] == args.scene]
    else:
        exit(1)
    
    if args.dataset == 'SceneFun3D':
        scene_pc = o3d.io.read_point_cloud(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.scene+'_laser_scan.ply')
        refined_transform = np.load(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/'+args.video+'_refined_transform.npy') 
        scene_pc.transform(refined_transform)
    elif args.dataset == 'FunGraph3D':
        scene_pc = o3d.io.read_point_cloud(args.root_path+'/'+args.scene+'/'+args.scene+'.ply')
    
    # if args.edge_file == '':
    #     if args.dataset == 'SceneFun3D':
    #         with open(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/cfslam_funcgraph_edges.pkl', "rb") as f:
    #             edges = pickle.load(f)
    #     elif args.dataset == 'FunGraph3D':
    #         with open(str(Path(args.root_path) / args.scene / args.video / args.gpt / 'cfslam_funcgraph_edges.pkl'), "rb") as f:
    #             edges = pickle.load(f)
    # else:
    #     with open(str(args.edge_file), "rb") as f:
    #         edges = pickle.load(f)

    if args.obj_file == '' or args.part_file == '':            
        if args.dataset == 'SceneFun3D':
            objects, _, _, _, _ = load_result(str(Path(args.root_path) / args.split / args.scene / args.video / args.gpt / 'full_pcd_ram_withbg_allclasses_overlap_maskconf0_updated.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz'))
            parts, _, _, _, _ = load_result(str(Path(args.root_path) / args.split / args.scene / args.video / args.gpt / 'full_pcd_ram_withbg_allclasses_overlap_maskconf0_updated.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz'))
        elif args.dataset == 'FunGraph3D':
            objects, _, _, _, _ = load_result(str(Path(args.root_path) / args.scene / args.video / args.gpt / 'full_pcd_ram_withbg_allclasses_overlap_maskconf0_updated.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz'))
            parts, _, _, _, _ = load_result(str(Path(args.root_path) / args.scene / args.video / args.gpt / 'full_pcd_ram_withbg_allclasses_overlap_maskconf0_updated.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz'))
    else:
        objects, _, _, _, _ = load_result(str(args.obj_file))
        # print(objects[0]['class_name'])
        # print(objects[0]['class_id'])
        print(objects[0]['refined_obj_tag'])
        parts, _, _, _, _ = load_result(str(args.part_file))    
        # print(parts[0]['class_name'])
        # print(parts[0]['class_id'])
        print(parts[0]['refined_obj_tag'])
    
    all_labels_embeddings = np.load(args.root_path+'/all_labels_clip_embeddings.npy')
    with open(args.root_path+'/all_labels.json', 'r') as f:
        all_labels = json.load(f)
    
    # with open(args.root_path+'/all_edges.refined.json', 'r') as f:
    #     all_edges = json.load(f)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    # sbert = SentenceTransformer("all-MiniLM-L6-v2")

    final_res_triplet = []
    final_res_node = []
    ks = [1, 2, 3, 5, 10]

    # ################################################### TRIPLET RECALL ####################################################
    # for k in ks:
    #     rk_num = 0
    #     rk_num_obj = 0
    #     rk_num_edge = 0
    #     fail_num = 0

    #     for gt_edge in tqdm(gt_edge_annos):
    #         try:
    #             gt_obj1 = [anno for anno in gt_annos if anno["annot_id"] == gt_edge["first_node_annot_id"]][0]
    #         except:
    #             fail_num += 1
    #             continue

    #         gt_label1 = gt_obj1['label']
    #         gt_mask1 = gt_obj1['indices']
    #         gt_pc1 = np.asarray(scene_pc.points)[gt_mask1]
    #         gt_pc1_o3d = o3d.geometry.PointCloud()
    #         gt_pc1_o3d.points = o3d.utility.Vector3dVector(gt_pc1)
    #         gt_bbd1 = gt_pc1_o3d.get_oriented_bounding_box()

    #         try:
    #             gt_obj2 = [anno for anno in gt_annos if anno["annot_id"] == gt_edge["second_node_annot_id"]][0]
    #         except:
    #             fail_num += 1
    #             continue

    #         gt_label2 = gt_obj2['label']
    #         gt_mask2 = gt_obj2['indices']
    #         gt_pc2 = np.asarray(scene_pc.points)[gt_mask2]
    #         gt_pc2_o3d = o3d.geometry.PointCloud()
    #         gt_pc2_o3d.points = o3d.utility.Vector3dVector(gt_pc2)
    #         gt_bbd2 = gt_pc2_o3d.get_oriented_bounding_box()

    #         gt_rel = gt_edge["description"]
    #         flag_obj = False
    #         flag_edge = False

    #         for edge in edges:
    #             if edge[2] == -1:
    #                 pred_func = parts[edge[1]]
    #             elif edge[1] == -1:
    #                 pred_func = objects[edge[0]]

    #             pred_bbd1 = pred_func['bbox']
    #             pred_label1 = pred_func['refined_obj_tag']
    #             iou1 = compute_3d_iou(gt_bbd1, pred_bbd1)
    #             if iou1 > args.iou_threshold:
    #                 inputs = processor(text=[pred_label1], return_tensors="pt", padding=True, truncation=True)
    #                 with torch.no_grad():
    #                     embeddings = model.get_text_features(**inputs)
    #                 embeddings = embeddings.numpy()
    #                 norm_embeddings = embeddings / np.linalg.norm(embeddings)
    #                 norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
    #                 similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
    #                 topk_indices = np.argsort(similarity[0], axis=0)[-k:][::-1]
    #                 topk_label = [all_labels[idx] for idx in topk_indices]
    #                 if args.debug: print(f'\nlabel1 - GT: {gt_label1} {pred_label1} iou: {iou1}')
    #                 if gt_label1 not in topk_label:
    #                     continue
    #             else:
    #                 continue

    #             if edge[2] == -1:
    #                 pred_obj = objects[edge[0]]
    #             elif edge[1] == -1:
    #                 pred_obj = objects[edge[2]]

    #             pred_bbd2 = pred_obj['bbox']
    #             pred_label2 = pred_obj['refined_obj_tag']
    #             iou2 = compute_3d_iou(gt_bbd2, pred_bbd2)
    #             if iou2 > args.iou_threshold:
    #                 inputs = processor(text=[pred_label2], return_tensors="pt", padding=True, truncation=True)
    #                 with torch.no_grad():
    #                     embeddings = model.get_text_features(**inputs)
    #                 embeddings = embeddings.numpy()
    #                 norm_embeddings = embeddings / np.linalg.norm(embeddings)
    #                 norm_all_labels_embeddings = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
    #                 similarity = np.dot(norm_embeddings, norm_all_labels_embeddings.T)
    #                 topk_indices = np.argsort(similarity[0], axis=0)[-k:][::-1]
    #                 topk_label = [all_labels[idx] for idx in topk_indices]
    #                 if args.debug: print(f'\nlabel2 - GT: {gt_label2} {pred_label2} iou: {iou2}')
    #                 if gt_label2 not in topk_label:
    #                     continue
    #             else:
    #                 continue

    #             flag_obj = True

    #             pred_rel = edge[3]
    #             embeddings = sbert.encode([pred_rel])
    #             gt_embeddings = sbert.encode(all_edges)
    #             similarities = sbert.similarity(embeddings, gt_embeddings)
    #             argsorted = sorted(
    #                 [(all_edges[t], similarities[0][t]) for t in range(len(all_edges))],
    #                 key=lambda x: x[1],
    #                 reverse=True,
    #             )
    #             topk_labels_rel = [lbl for lbl, _ in argsorted[:k]]
    #             if args.debug:
    #                 print(f'\nGT: {gt_rel}')
    #                 print(f'Pred rel: {pred_rel}')
    #                 print(f'Predicted topk edges: {argsorted[:k]}')
    #             if gt_rel in topk_labels_rel:
    #                 if args.debug: print('Correctly retrieved edge!!!!')
    #                 flag_edge = True
    #                 rk_num += 1
    #                 break

    #         if flag_obj:
    #             rk_num_obj += 1
    #         if flag_edge:
    #             rk_num_edge += 1

    #     if args.debug:
    #         print(f'Top {k} Triplet Recall: {rk_num} / {len(gt_edge_annos) - fail_num} = {rk_num / (len(gt_edge_annos) - fail_num):.4f}')
    #         print(f'Top {k} Object Recall: {rk_num_obj} / {len(gt_edge_annos) - fail_num} = {rk_num_obj / (len(gt_edge_annos) - fail_num):.4f}')
    #         if rk_num_obj != 0:
    #             print(f'Top {k} Edge Recall: {rk_num_edge} / {rk_num_obj} = {rk_num_edge / rk_num_obj:.4f}')
    #         else:
    #             print(f'Top {k} Edge Recall: {rk_num_edge} / {rk_num_obj} = 0.0000')

    #     final_res_triplet.extend([len(gt_edge_annos) - fail_num, rk_num_obj, rk_num_edge, rk_num])

    ################################################### NODE RECALL ####################################################
    obj_idx = []
    part_idx = []
    # for e in edges:
    #     if e[2] == -1:
    #         obj_idx.append(e[0]); part_idx.append(e[1])
    #     elif e[1] == -1:
    #         obj_idx.append(e[0]); obj_idx.append(e[2])
    for o in objects:
        # obj_idx.append(o['refined_obj_tag'])
        # obj_idx.append(_majority_id(o['class_id'])) # class id가 아니라 MOL에서의 index 줘야함
        obj_idx.append(objects.index(o))
    for p in parts:
        # part_idx.append(p['refined_obj_tag'])
        # part_idx.append(_majority_id(p['class_id']))
        part_idx.append(parts.index(p))

    obj_idx = list(set(obj_idx))
    part_idx = list(set(part_idx))

    total_obj_num = 0
    total_func_num = 0
    for gt in gt_annos:
        if gt['label'] in CLASS_LABELS_FUNC:
            total_func_num += 1
        else:
            total_obj_num += 1

    for k in ks:
        r_obj = 0
        r_func = 0
        for gt in tqdm(gt_annos, desc=f'Node R@{k}'):
            gt_label = gt['label']
            gt_mask = gt['indices']
            gt_pc = np.asarray(scene_pc.points)[gt_mask]
            gt_pc_o3d = o3d.geometry.PointCloud()
            gt_pc_o3d.points = o3d.utility.Vector3dVector(gt_pc)
            gt_bbd = gt_pc_o3d.get_oriented_bounding_box()

            found = False

            for obj_id in obj_idx:
                pred = objects[obj_id]
                iou = compute_3d_iou(gt_bbd, pred['bbox'])
                if iou <= args.iou_threshold:
                    continue
                pred_label = pred['refined_obj_tag']
                inputs = processor(text=[pred_label], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    emb = model.get_text_features(**inputs).detach().cpu().numpy()
                norm_emb = emb / np.linalg.norm(emb)
                norm_all = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                sim = np.dot(norm_emb, norm_all.T)
                topk_idx = np.argsort(sim[0])[-k:][::-1]
                topk_labels = [all_labels[idx] for idx in topk_idx]
                if gt_label in topk_labels:
                    if gt_label in CLASS_LABELS_FUNC:
                        r_func += 1
                    else:
                        r_obj += 1
                    found = True
                    break

            if found:
                continue

            for part_id in part_idx:
                pred = parts[part_id]
                iou = compute_3d_iou(gt_bbd, pred['bbox'])
                if iou <= args.iou_threshold:
                    continue
                pred_label = pred['refined_obj_tag']
                inputs = processor(text=[pred_label], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    emb = model.get_text_features(**inputs).detach().cpu().numpy()
                norm_emb = emb / np.linalg.norm(emb)
                norm_all = all_labels_embeddings / np.linalg.norm(all_labels_embeddings, axis=1, keepdims=True)
                sim = np.dot(norm_emb, norm_all.T)
                topk_idx = np.argsort(sim[0])[-k:][::-1]
                topk_labels = [all_labels[idx] for idx in topk_idx]
                if gt_label in topk_labels:
                    if gt_label in CLASS_LABELS_FUNC:
                        r_func += 1
                    else:
                        r_obj += 1
                    break

        obj_recall = (r_obj / total_obj_num) if total_obj_num else 0.0
        func_recall = (r_func / total_func_num) if total_func_num else 0.0
        overall_recall = ((r_obj + r_func) / (total_obj_num + total_func_num)) if (total_obj_num + total_func_num) else 0.0
        
        if args.debug:
            print(f'Top {k} Object-node Recall: {r_obj} / {total_obj_num} = {obj_recall:.4f}')
            print(f'Top {k} Part-node Recall: {r_func} / {total_func_num} = {func_recall:.4f}')
            print(f'Top {k} Overall-node Recall: {overall_recall:.4f}')

        final_res_node.extend([total_obj_num, total_func_num, r_obj, r_func, r_obj + r_func])
    
    print("\n\n")
    print(f"Triplet Summary: GT | Assoc.Node | Edge | Triplet (for Top k = {ks})")
    for i in range(len(ks)):
        print(f"Top {ks[i]}: ", final_res_triplet[i*4:(i+1)*4])

    print(f"Node Summary: GT Obj | GT Part | Obj.Node | Part.Node | Overall Node (for Top k = {ks})")
    for i in range(len(ks)):
        print(f"Top {ks[i]}: ", final_res_node[i*5:(i+1)*5])

    print("\n-------------------------------------------")
    print(final_res_node[2:4], final_res_node[12:14], final_res_node[17:19], final_res_node[22:24])
    print(final_res_triplet[1:4], final_res_triplet[9:12], final_res_triplet[13:16], final_res_triplet[17:20])
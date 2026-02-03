#!/usr/bin/env python3
"""
Evaluation code for FunGraph3D / SceneFun3D functional 3D scene graph prediction.

This script evaluates (label-space) recall@k for:
  - OBJECT node
  - PART node (functional part node; determined by CLASS_LABELS_FUNC)
  - overall node (= object + part)
  - Assoc. Node (functional relation node-pair match only)
  - FUNCTIONAL triplet (functional relation node-pair + relation label)
  - SPATIAL triplet (spatial relation node-pair; relation label is always "part of")
  - AFFORDANCE (part node + affordance label)

Key points:
  - Requires --obj_file and --part_file (pkl.gz with bboxes).
  - Requires --graph_file (json) which contains predicted node labels, affordances, and relations.
  - Does NOT use --edge_file.
  - Computes recall for k in {1,2,3,5,10}.
  - Uses CLIP for node label-space retrieval and SBERT for functional relation labels and affordances.
  - For spatial relations, label-space is only "part of", so only node matching is evaluated.
  - Functional/spatial relation node pairs can be (obj,obj), (part,part), or (obj,part). Order is ignored.

Dependencies:
  pip install "transformers==4.35.2" open3d sentence-transformers==5.1.2 tqdm
"""
import argparse
import gzip
import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------- Do not change the following label lists ----------------------- #
# Focused functional part classes in SceneFun3D / FunGraph3D dataset
CLASS_LABELS_FUNC = [
    "button / knob",  "power strip", "light switch", "faucet / handle", "button", "handle", "knob", "knob / button",
    "faucet / knob / handle", "switch panel / electric outlet", "remote", "electric outlet / power strip",
    "handle / faucet", "switch panel", "electric outlet"
]

# SceneFun3D_Graph original labels
all_nodes_scenefun3d = [
    "handle / faucet", "washing machine", "handle", "bathtub", "bathroom sink", "dryer", "light bulb", 
    "window", "fridge", "light switch", "wardrobe", "dresser / nightstand", "toilet", "trashcan", "knob / button", 
    "door", "laptop", "sink", "faucet / handle", "television stand / cabinet", "cabinet / closet", "lamp", "remote", 
    "radiator", "drawer", "glass door", "chest of drawers / dresser", "kettle", "nightstand / dresser", "projector", 
    "dresser / chest of drawers", "kitchen sink", "power strip", "television", "kitchen cabinet", "faucet / knob / handle", 
    "oven", "cabinet / dresser / nightstand", "exhaust hood / ventilation fan", "switch panel", "knob", "dishwasher", 
    "doors", "nightstand drawer", "chandelier / ceiling light", "switch panel / electric outlet", "ceiling light fixture", 
    "cabinet", "electric outlet", "button / knob", "ceiling light", "button", "electric outlet / power strip", "drawer / cabinet"
    ]

# FunGraph3D original labels
all_nodes_fungraph3d = [
    "faucet / handle", "toilet", "electric outlet", "ceiling light fixture", "dishwasher button / dishwasher switch", 
    "coffee machine", "handle", "cooker", "electric outlet / switch panel", "trashcan", "window", "radiator", "coffee maker", 
    "kitchen cabinet / fridge", "desk drawer", "fan", "bathroom sink", "dishwasher", "button", "kitchen cabinet", "cabinet", 
    "microwave oven", "switch / electric outlet", "power strip / electric outlet", "electric cooker", "toaster", "television", 
    "knob / handle / faucet", "wardrobe", "kitchen cabinet / drawer", "switch panel / electric outlet", "switch panel", 
    "light fixture", "exhaust hood", "kitchen cabinet / kitchen counter", "fridge", "button / knob", "oven", "lamp", 
    "dresser / chest of drawers", "handle / knob", "bathtub", "door", "switch", "power strip", "kitchen sink", "bathroom vanity", 
    "kettle", "knob", "drawer / kitchen cabinet"
    ]

# SceneFun3D divided & refined labels
all_nodes_scenefun3d_divided = [
    "bathroom sink", "bathtub", "button", "cabinet", "ceiling light", "ceiling light fixture", "chandelier",
    "closet", "dishwasher", "door", "drawer", "dresser", "dryer", "electric outlet", "exhaust hood",
    "faucet", "fridge", "glass door", "handle", "kettle", "kitchen cabinet", "kitchen sink", "knob", "lamp",
    "laptop", "light bulb", "light switch", "nightstand", "oven", "power strip", "projector", "radiator", "remote",
    "sink", "switch panel", "television", "television stand", "toilet", "trashcan", "ventilation fan", "wardrobe",
    "washing machine", "window"
]

# FunGraph3D divided labels
all_nodes_fungraph3d_divided = [
    "bathroom sink", "bathroom vanity", "bathtub", "button", "cabinet", "ceiling light fixture", "coffee machine",
    "coffee maker", "cooker", "desk drawer", "dishwasher", "dishwasher button", "dishwasher switch", "door", "drawer",
    "dresser", "electric cooker", "electric outlet", "exhaust hood", "fan", "faucet", "fridge", "handle", "kettle",
    "kitchen cabinet", "kitchen counter", "kitchen sink", "knob", "lamp", "light fixture", "microwave oven", "oven",
    "power strip", "radiator", "switch", "switch panel", "television", "toaster", "toilet", "trashcan", "wardrobe", "window"
]
# ----------------------------------------------------------------------- #


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["SceneFun3D", "FunGraph3D"])
    p.add_argument("--root_path", type=str, required=True)
    p.add_argument("--scene", type=str, required=True)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--split", type=str, default=None)

    p.add_argument("--bbox_threshold", type=float, default=0.0, help="BBox match threshold (interpreted as IoU/IoP depending on --bbox_match).")
    p.add_argument("--bbox_match", type=str, default="iou", choices=["iou", "iop"], help="BBox match metric: iou or iop.")

    p.add_argument("--obj_file", type=str, required=True)
    p.add_argument("--part_file", type=str, required=True)
    p.add_argument("--graph_file", type=str, required=True)

    p.add_argument("--debug", action="store_true")
    return p


def normalize_text(s: str) -> str:
    """Normalize text for embedding (remove/space-out specific symbols)."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.replace(":", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_3d_iou(bbox1: o3d.geometry.OrientedBoundingBox, bbox2: o3d.geometry.OrientedBoundingBox) -> float:
    """
    Axis-aligned IoU computed from bbox min/max bounds.
    """
    bbox1_min = np.asarray(bbox1.get_min_bound())
    bbox1_max = np.asarray(bbox1.get_max_bound())
    bbox2_min = np.asarray(bbox2.get_min_bound())
    bbox2_max = np.asarray(bbox2.get_max_bound())

    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = float(np.prod(overlap_size))
    v1 = float(np.prod(bbox1_max - bbox1_min))
    v2 = float(np.prod(bbox2_max - bbox2_min))
    union = v1 + v2 - overlap_volume
    if union <= 0:
        return 0.0
    return overlap_volume / union

def compute_3d_iop(bbox1: o3d.geometry.OrientedBoundingBox, bbox2: o3d.geometry.OrientedBoundingBox) -> float:
    """
    Axis-aligned IoP computed from bbox min/max bounds.
    """
    bbox1_min = np.asarray(bbox1.get_min_bound())
    bbox1_max = np.asarray(bbox1.get_max_bound())
    bbox2_min = np.asarray(bbox2.get_min_bound())
    bbox2_max = np.asarray(bbox2.get_max_bound())

    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = float(np.prod(overlap_size))
    v1 = float(np.prod(bbox1_max - bbox1_min))
    v2 = float(np.prod(bbox2_max - bbox2_min))
    if v2 <= 0:
        return 0.0
    return overlap_volume / v2


def load_map_objects(result_path: str):
    """
    Load pkl.gz result (object/part) and reconstruct Open3D bboxes.

    The file is expected to contain a dict with key "objects",
    which is a list of serializable detection dicts including 'bbox_np'.
    """
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    if "objects" not in results:
        raise KeyError(f'"objects" key not found in: {result_path}')
    s_obj_list = results["objects"]

    out = []
    for s_obj_dict in s_obj_list:
        if "bbox_np" not in s_obj_dict:
            raise KeyError(f'"bbox_np" key not found in detection dict in: {result_path}')
        bbox_np = np.asarray(s_obj_dict["bbox_np"])
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_np))
        out.append({"bbox": bbox})
    return out


def parse_graph_pair(pair):
    """
    Parse graph relation pair, allowing any of:
      (obj, obj), (obj, part), (part, obj), (part, part)

    Returns: (t1, i1, t2, i2) where t in {"obj","part"} and i is int index.
    """
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"pair must be a list/tuple of length 2, got: {pair}")

    def _parse_one(s: str):
        if not isinstance(s, str):
            raise ValueError(f"pair entries must be str, got: {type(s)} / {s}")
        if s.startswith("obj_"):
            return "obj", int(s.split("_", 1)[1])
        if s.startswith("part_"):
            return "part", int(s.split("_", 1)[1])
        raise ValueError(f"pair entry must start with obj_ or part_, got: {s}")

    t1, i1 = _parse_one(pair[0])
    t2, i2 = _parse_one(pair[1])
    return t1, i1, t2, i2


def build_clip_label_cache(model, processor, norm_all_node_embed, all_nodes, max_k: int):
    """
    Returns a function get_top_labels(text) -> list[str] of length max_k.
    Uses an internal cache to avoid redundant CLIP forward passes.
    """
    cache = {}

    def get_top_labels(text: str):
        text_norm = normalize_text(text)
        if text_norm in cache:
            return cache[text_norm]
        inputs = processor(text=[text_norm], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model.get_text_features(**inputs).detach().cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        sim = np.dot(emb, norm_all_node_embed.T)[0]  # (N,)
        topk_idx = np.argsort(sim)[-max_k:][::-1]
        topk_labels = [all_nodes[i] for i in topk_idx]
        cache[text_norm] = topk_labels
        return topk_labels

    return get_top_labels

def build_sbert_label_cache(sbert: SentenceTransformer, label_texts_original, label_emb, max_k: int):
    """
    Returns get_top_labels(text) -> list[str] (length <= max_k), in ORIGINAL label strings.
    Uses normalized query text for embedding. Handles label spaces smaller than max_k.
    """
    cache = {}

    num_labels = int(label_emb.shape[0])
    if num_labels <= 0:
        raise ValueError("Label space is empty (num_labels=0)")

    effective_k = min(int(max_k), num_labels)

    def get_top_labels(text: str):
        text_norm = normalize_text(text)
        if text_norm in cache:
            return cache[text_norm]
        q_emb = sbert.encode([text_norm], convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(q_emb, label_emb)[0]  # (num_labels,)
        topk_idx = torch.topk(sim, k=effective_k, largest=True).indices.tolist()
        topk_labels = [label_texts_original[idx] for idx in topk_idx]
        cache[text_norm] = topk_labels
        return topk_labels

    return get_top_labels

def rank_of_gt_in_topk(gt_label: str, topk_labels: list[str], max_k: int) -> int:
    """
    Return 1-based rank of gt_label (or any token split by '/') in topk_labels.
    If not found within the provided list, return max_k+1 (so it never counts as a hit).

    Note: topk_labels length may be < max_k when the label space itself is smaller.
    """
    if gt_label in topk_labels:
        return topk_labels.index(gt_label) + 1

    for token in str(gt_label).split("/"):
        token = token.strip()
        if not token:
            continue
        if token in topk_labels:
            return topk_labels.index(token) + 1

    return int(max_k) + 1

def match_rank(
    gt_bbox,
    gt_label,
    pred_bbox,
    pred_label,
    get_top_node_labels,
    max_k: int,
    bbox_th: float,
    overlap_fn,
    overlap_name: str = "IoU",
    debug: bool = False,
    debug_prefix: str = "",
) -> int:
    """
    Returns the minimal rank r (1..max_k) such that:
      overlap(gt_bbox, pred_bbox) > bbox_th AND gt_label in top-r labels of pred_label
    """
    score = overlap_fn(gt_bbox, pred_bbox)
    if score <= bbox_th:
        return max_k + 1
    topk = get_top_node_labels(pred_label)
    r = rank_of_gt_in_topk(gt_label, topk, max_k=max_k)
    if debug:
        print(
            f"{debug_prefix} {overlap_name}={score:.4f} pred_label='{pred_label}' "
            f"top{max_k}={topk} | gt_label='{gt_label}' rank={r}"
        )
    return r


def unordered_pair_rank_any(
    gt1,
    gt2,
    pred1,
    pred2,
    get_top_node_labels,
    max_k: int,
    bbox_th: float,
    overlap_fn,
    overlap_name: str = "IoU",
    debug: bool = False,
    debug_prefix: str = "",
) -> int:
    """
    Compute best rank for unordered node-pair match between:
      GT nodes (gt1, gt2) and predicted nodes (pred1, pred2)

    Pair matches at k iff both endpoints match within k (IoU + label-space).
    We allow both orientations and return the best (minimum) required k.
    """
    # orientation A: pred1->gt1, pred2->gt2
    r_a1 = match_rank(
        gt1["bbox"], gt1["label"], pred1["bbox"], pred1["label"],
        get_top_node_labels, max_k, bbox_th, overlap_fn, overlap_name, debug=debug, debug_prefix=debug_prefix + " [A-gt1<-pred1]"
    )
    r_a2 = match_rank(
        gt2["bbox"], gt2["label"], pred2["bbox"], pred2["label"],
        get_top_node_labels, max_k, bbox_th, overlap_fn, overlap_name, debug=debug, debug_prefix=debug_prefix + " [A-gt2<-pred2]"
    )
    rank_a = max(r_a1, r_a2)

    # orientation B: pred2->gt1, pred1->gt2
    r_b1 = match_rank(
        gt1["bbox"], gt1["label"], pred2["bbox"], pred2["label"],
        get_top_node_labels, max_k, bbox_th, overlap_fn, overlap_name, debug=debug, debug_prefix=debug_prefix + " [B-gt1<-pred2]"
    )
    r_b2 = match_rank(
        gt2["bbox"], gt2["label"], pred1["bbox"], pred1["label"],
        get_top_node_labels, max_k, bbox_th, overlap_fn, overlap_name, debug=debug, debug_prefix=debug_prefix + " [B-gt2<-pred1]"
    )
    rank_b = max(r_b1, r_b2)

    return min(rank_a, rank_b)


def print_summary_table(gt_counts, metrics_counts, ks):
    """
    Print a wide summary table similar to the provided screenshot.
    metrics_counts: dict[str, dict[int,int]]
      e.g., metrics_counts["OBJECT node"][1] = hit_count
    """
    groups = [
        "OBJECT node",
        "PART node",
        "overall node",
        "Assoc. Node",
        "FUNCTIONAL triplet",
        "SPATIAL triplet",
        "AFFORDANCE",
        "only AFFORDANCE",
    ]
    k_cols = [f"R@{k}" for k in ks]

    left_cols = ["Obj", "Part", "Func", "Spat", "Affor", "onlyAffor"]
    left_header = "GT".ljust(14) + " ".join([c.rjust(6) for c in left_cols])

    group_header = left_header + " | "
    for g in groups:
        width = 6 * len(k_cols) + (len(k_cols) - 1)
        group_header += g.center(width) + " | "

    sub_header = " ".ljust(len(left_header)) + " | "
    for _ in groups:
        sub_header += " ".join([c.rjust(6) for c in k_cols]) + " | "

    gt_line = "GT".ljust(14) + " ".join([str(gt_counts[c]).rjust(6) for c in left_cols])
    val_line = gt_line + " | "
    for g in groups:
        val_line += " ".join([str(metrics_counts[g][k]).rjust(6) for k in ks]) + " | "

    print("\n" + "-" * len(val_line))
    print(group_header)
    print(sub_header)
    print(val_line)
    print("-" * len(val_line) + "\n")


def main():
    args = get_parser().parse_args()
    root = Path(args.root_path)

    # Choose bbox overlap metric function
    if args.bbox_match == "iou":
        overlap_fn = compute_3d_iou
        overlap_name = "IoU"
    elif args.bbox_match == "iop":
        overlap_fn = compute_3d_iop
        overlap_name = "IoP"
    else:
        raise ValueError(f"Unknown bbox_match: {args.bbox_match}")

    bbox_th = float(args.bbox_threshold)
    print(f"[BBoxMatch] metric={args.bbox_match} ({overlap_name}), threshold={bbox_th}")

    # ---------------------- Load GT files ---------------------- #
    prefix = args.dataset  # "SceneFun3D" or "FunGraph3D"

    # NOTE: keep explicit path (no fallback logic). Adjust this path in your environment if needed.
    GT_path = Path("/home/main/workspace/k2room2/CAPA-3DSG/dataset/UniGraph3D")
    nodes_path = GT_path / f"{prefix}.nodes.json"
    func_rel_path = GT_path / f"{prefix}.relations.functional.json"
    spat_rel_path = GT_path / f"{prefix}.relations.spatial.json"
    edge_labels_path = GT_path / f"{prefix}.labels.edge.json"
    aff_labels_path = GT_path / f"{prefix}.labels.affordance.json"

    for p in [nodes_path, func_rel_path, spat_rel_path, edge_labels_path, aff_labels_path]:
        if not p.exists():
            raise FileNotFoundError(f"GT file not found: {p}")

    with open(nodes_path, "r") as f:
        gt_nodes_all = json.load(f)
    gt_nodes = [n for n in gt_nodes_all if n.get("scene_id") == args.scene]

    with open(func_rel_path, "r") as f:
        gt_func_all = json.load(f)
    gt_func = [r for r in gt_func_all if r.get("scene_id") == args.scene]

    with open(spat_rel_path, "r") as f:
        gt_spat_all = json.load(f)
    gt_spat = [r for r in gt_spat_all if r.get("scene_id") == args.scene]

    with open(edge_labels_path, "r") as f:
        edge_labels_original = json.load(f)
    with open(aff_labels_path, "r") as f:
        aff_labels_original = json.load(f)

    # ---------------------- Load scene point cloud ---------------------- #
    if args.dataset == 'SceneFun3D':
        scene_pc = o3d.io.read_point_cloud(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.scene+'_laser_scan.ply')
        refined_transform = np.load(args.root_path+'/'+args.split+'/'+args.scene+'/'+args.video+'/'+args.video+'_refined_transform.npy') 
        scene_pc.transform(refined_transform)
    elif args.dataset == 'FunGraph3D':
        scene_pc = o3d.io.read_point_cloud(args.root_path+'/'+args.scene+'/'+args.scene+'.ply')

    scene_pts = np.asarray(scene_pc.points)

    # ---------------------- Load predictions (obj/part + graph) ---------------------- #
    print("Loading predicted object/part files...")
    objects_bbox_only = load_map_objects(args.obj_file)
    parts_bbox_only = load_map_objects(args.part_file)
    print(f"  objects: {len(objects_bbox_only)}  parts: {len(parts_bbox_only)}")

    with open(args.graph_file, "r") as f:
        graph = json.load(f)

    required_graph_keys = ["object", "part", "functional_relation", "spatial_relation"]
    for k in required_graph_keys:
        if k not in graph:
            raise KeyError(f"graph_file must contain key '{k}'")

    # Pred node ids and labels from graph_file
    pred_obj_ids = sorted([int(k.split("_", 1)[1]) for k in graph["object"].keys() if k.startswith("obj_")])
    pred_part_ids = sorted([int(k.split("_", 1)[1]) for k in graph["part"].keys() if k.startswith("part_")])

    # Validate indices against bbox lists
    if pred_obj_ids and (max(pred_obj_ids) >= len(objects_bbox_only)):
        raise IndexError(
            f"graph_file references obj_{max(pred_obj_ids)} but obj_file has only {len(objects_bbox_only)} objects"
        )
    if pred_part_ids and (max(pred_part_ids) >= len(parts_bbox_only)):
        raise IndexError(
            f"graph_file references part_{max(pred_part_ids)} but part_file has only {len(parts_bbox_only)} parts"
        )

    pred_func_rel = graph["functional_relation"]
    if not isinstance(pred_func_rel, list):
        raise TypeError("graph['functional_relation'] must be a list")

    pred_spat_rel = graph["spatial_relation"]
    if not isinstance(pred_spat_rel, list):
        raise TypeError("graph['spatial_relation'] must be a list")

    # ---------------------- Build GT node index & bboxes ---------------------- #
    func_label_set = set(normalize_text(x) for x in CLASS_LABELS_FUNC)

    gt_node_by_annot = {}
    gt_obj_count = 0
    gt_part_count = 0
    gt_aff_instances = []  # list of dicts: {"bbox","label","aff_label"}

    print("Precomputing GT node bounding boxes...")
    for n in tqdm(gt_nodes, desc="GT nodes"):
        annot_id = n.get("annot_id")
        if annot_id is None:
            continue
        label = n.get("label", "")
        indices = n.get("indices", [])
        if not isinstance(indices, list) or len(indices) == 0:
            continue

        pts = scene_pts[np.asarray(indices, dtype=np.int64)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        bbox = pcd.get_oriented_bounding_box()

        is_part = normalize_text(label) in func_label_set
        if is_part:
            gt_part_count += 1
        else:
            gt_obj_count += 1

        aff_list = n.get("affordance", [])
        if not isinstance(aff_list, list):
            raise TypeError("GT node 'affordance' must be a list")

        gt_node_by_annot[annot_id] = {
            "bbox": bbox,
            "label": label,
            "is_part": is_part,
            "affordance": aff_list,
        }

        # Affordance instances (multi-label allowed)
        if is_part:
            for a in aff_list:
                gt_aff_instances.append({"bbox": bbox, "label": label, "aff_label": a})

    gt_func_count = len(gt_func)
    gt_spat_count = len(gt_spat)
    gt_aff_count = len(gt_aff_instances)

    # ---------------------- Load models & precompute label embeddings ---------------------- #
    ks = [1, 2, 3, 5, 10]
    max_k = max(ks)

    print("Loading CLIP and SBERT models...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    if args.dataset == "SceneFun3D":
        all_nodes = all_nodes_scenefun3d_divided
    else:
        all_nodes = all_nodes_fungraph3d_divided

    print("Computing node label embeddings (CLIP)...")
    all_node_embed = []
    for lbl in tqdm(all_nodes, desc="Node label emb"):
        lbl_norm = normalize_text(lbl)
        inputs = clip_processor(text=[lbl_norm], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            e = clip_model.get_text_features(**inputs).detach().cpu().numpy()[0]
        all_node_embed.append(e)
    all_node_embed = np.stack(all_node_embed, axis=0)
    norm_all_node_embed = all_node_embed / (np.linalg.norm(all_node_embed, axis=1, keepdims=True) + 1e-12)

    get_top_node_labels = build_clip_label_cache(
        clip_model, clip_processor, norm_all_node_embed, all_nodes, max_k=max_k
    )

    print("Computing functional edge label embeddings (SBERT)...")
    edge_labels_norm = [normalize_text(x) for x in edge_labels_original]
    edge_label_emb = sbert.encode(edge_labels_norm, convert_to_tensor=True, show_progress_bar=True)
    get_top_edge_labels = build_sbert_label_cache(sbert, edge_labels_original, edge_label_emb, max_k=max_k)

    print("Computing affordance label embeddings (SBERT)...")
    aff_labels_norm = [normalize_text(x) for x in aff_labels_original]
    aff_label_emb = sbert.encode(aff_labels_norm, convert_to_tensor=True, show_progress_bar=True)
    get_top_aff_labels = build_sbert_label_cache(sbert, aff_labels_original, aff_label_emb, max_k=max_k)

    # ---------------------- Prepare predicted node dicts ---------------------- #
    pred_objects = {}
    for obj_id in pred_obj_ids:
        key = f"obj_{obj_id}"
        # strict: require label
        pred_objects[obj_id] = {
            "bbox": objects_bbox_only[obj_id]["bbox"],
            "label": graph["object"][key]["label"],
        }

    pred_parts = {}
    for part_id in pred_part_ids:
        key = f"part_{part_id}"
        pred_parts[part_id] = {
            "bbox": parts_bbox_only[part_id]["bbox"],
            "label": graph["part"][key]["label"],
            "affordance": graph["part"][key]["affordance"],
        }
        if not isinstance(pred_parts[part_id]["affordance"], list):
            raise TypeError("Predicted part 'affordance' must be a list")

    def get_pred_node(kind: str, idx: int):
        """Return predicted node dict with bbox+label. Skip if the referenced node is missing."""
        if kind == "obj":
            return pred_objects.get(idx, None)
        if kind == "part":
            return pred_parts.get(idx, None)
        raise ValueError(kind)

    all_pred_nodes = list(pred_objects.values()) + list(pred_parts.values())

    # ---------------------- Evaluate metrics ---------------------- #
    metrics = {
        "OBJECT node": {k: 0 for k in ks},
        "PART node": {k: 0 for k in ks},
        "overall node": {k: 0 for k in ks},
        "Assoc. Node": {k: 0 for k in ks},
        "FUNCTIONAL triplet": {k: 0 for k in ks},
        "SPATIAL triplet": {k: 0 for k in ks},
        "AFFORDANCE": {k: 0 for k in ks},
        "only AFFORDANCE": {k: 0 for k in ks},
    }

    # 1) Node recall (OBJECT / PART / overall)
    print("Evaluating node recall...")
    for annot_id, gt in tqdm(gt_node_by_annot.items(), desc="Node eval"):
        best_rank = max_k + 1
        for pred in all_pred_nodes:
            r = match_rank(
                gt["bbox"],
                gt["label"],
                pred["bbox"],
                pred["label"],
                get_top_node_labels,
                max_k=max_k,
                bbox_th=bbox_th,
                overlap_fn=overlap_fn,
                overlap_name=overlap_name,
                debug=args.debug,
                debug_prefix=f"[Node annot={annot_id}]",
            )
            if r < best_rank:
                best_rank = r
                if best_rank == 1:
                    break

        for k in ks:
            if best_rank <= k:
                if gt["is_part"]:
                    metrics["PART node"][k] += 1
                else:
                    metrics["OBJECT node"][k] += 1

    for k in ks:
        metrics["overall node"][k] = metrics["OBJECT node"][k] + metrics["PART node"][k]

    def get_gt_node(annot_id: str):
        return gt_node_by_annot.get(annot_id, None)

    # 2) Functional relations: Assoc. Node and Functional triplet
    print("Evaluating functional relations (Assoc. Node / FUNCTIONAL triplet)...")
    fail_func = 0
    for rel in tqdm(gt_func, desc="Functional rel eval"):
        id1 = rel.get("first_node_annot_id")
        id2 = rel.get("second_node_annot_id")
        gt1 = get_gt_node(id1)
        gt2 = get_gt_node(id2)
        if gt1 is None or gt2 is None:
            fail_func += 1
            continue

        gt_rel_label = rel.get("description", "")

        best_assoc_rank = max_k + 1
        best_triplet_rank = max_k + 1

        for pred_rel in pred_func_rel:
            if "pair" not in pred_rel or "label" not in pred_rel:
                raise KeyError("Each item in graph['functional_relation'] must have keys: 'pair', 'label'")

            t1, i1, t2, i2 = parse_graph_pair(pred_rel["pair"])
            pred1 = get_pred_node(t1, i1)
            pred2 = get_pred_node(t2, i2)
            if pred1 is None or pred2 is None:
                # skip invalid relation referencing missing node ids
                continue

            nodepair_rank = unordered_pair_rank_any(
                gt1={"bbox": gt1["bbox"], "label": gt1["label"]},
                gt2={"bbox": gt2["bbox"], "label": gt2["label"]},
                pred1={"bbox": pred1["bbox"], "label": pred1["label"]},
                pred2={"bbox": pred2["bbox"], "label": pred2["label"]},
                get_top_node_labels=get_top_node_labels,
                max_k=max_k,
                bbox_th=bbox_th,
                overlap_fn=overlap_fn,
                overlap_name=overlap_name,
                debug=args.debug,
                debug_prefix="[FuncRel]",
            )
            if nodepair_rank < best_assoc_rank:
                best_assoc_rank = nodepair_rank

            # relation label rank (SBERT label-space)
            pred_rel_text = pred_rel["label"]
            top_edge_labels = get_top_edge_labels(pred_rel_text)
            rel_rank = rank_of_gt_in_topk(gt_rel_label, top_edge_labels, max_k=max_k)

            triplet_rank = max(nodepair_rank, rel_rank)
            if triplet_rank < best_triplet_rank:
                best_triplet_rank = triplet_rank
                if best_triplet_rank == 1:
                    break

        for k in ks:
            if best_assoc_rank <= k:
                metrics["Assoc. Node"][k] += 1
            if best_triplet_rank <= k:
                metrics["FUNCTIONAL triplet"][k] += 1

    # 3) Spatial triplet (only "part of" label; evaluate node pair)
    print("Evaluating spatial relations (SPATIAL triplet)...")
    fail_spat = 0
    for rel in tqdm(gt_spat, desc="Spatial rel eval"):
        id1 = rel.get("first_node_annot_id")
        id2 = rel.get("second_node_annot_id")
        gt1 = get_gt_node(id1)
        gt2 = get_gt_node(id2)
        if gt1 is None or gt2 is None:
            fail_spat += 1
            continue

        best_rank = max_k + 1
        for pred_rel in pred_spat_rel:
            if "pair" not in pred_rel:
                raise KeyError("Each item in graph['spatial_relation'] must have key: 'pair'")

            t1, i1, t2, i2 = parse_graph_pair(pred_rel["pair"])
            pred1 = get_pred_node(t1, i1)
            pred2 = get_pred_node(t2, i2)
            if pred1 is None or pred2 is None:
                continue

            nodepair_rank = unordered_pair_rank_any(
                gt1={"bbox": gt1["bbox"], "label": gt1["label"]},
                gt2={"bbox": gt2["bbox"], "label": gt2["label"]},
                pred1={"bbox": pred1["bbox"], "label": pred1["label"]},
                pred2={"bbox": pred2["bbox"], "label": pred2["label"]},
                get_top_node_labels=get_top_node_labels,
                max_k=max_k,
                bbox_th=bbox_th,
                overlap_fn=overlap_fn,
                overlap_name=overlap_name,
                debug=args.debug,
                debug_prefix="[SpatRel]",
            )
            if nodepair_rank < best_rank:
                best_rank = nodepair_rank
                if best_rank == 1:
                    break

        for k in ks:
            if best_rank <= k:
                metrics["SPATIAL triplet"][k] += 1

    # 4) Affordance recall (GT part + affordance label)
    print("Evaluating affordance (AFFORDANCE)...")
    for gt_aff in tqdm(gt_aff_instances, desc="Affordance eval"):
        gt_bbox = gt_aff["bbox"]
        gt_label = gt_aff["label"]
        gt_aff_label = gt_aff["aff_label"]

        best_rank = max_k + 1
        # candidates: predicted parts only (only parts have affordance field)
        for pred_part in pred_parts.values():
            node_rank = match_rank(
                gt_bbox,
                gt_label,
                pred_part["bbox"],
                pred_part["label"],
                get_top_node_labels,
                max_k=max_k,
                bbox_th=bbox_th,
                overlap_fn=overlap_fn,
                overlap_name=overlap_name,
                debug=args.debug,
                debug_prefix="[Aff-node]",
            )
            if node_rank > max_k:
                continue

            pred_aff_list = pred_part.get("affordance", [])
            if not isinstance(pred_aff_list, list) or len(pred_aff_list) == 0:
                continue

            best_aff_label_rank = max_k + 1
            for pred_aff in pred_aff_list:
                top_aff_labels = get_top_aff_labels(pred_aff)
                r = rank_of_gt_in_topk(gt_aff_label, top_aff_labels, max_k=max_k)
                if r < best_aff_label_rank:
                    best_aff_label_rank = r

            cand_rank = max(node_rank, best_aff_label_rank)
            if cand_rank < best_rank:
                best_rank = cand_rank
                if best_rank == 1:
                    break

        for k in ks:
            if best_rank <= k:
                metrics["AFFORDANCE"][k] += 1

    # 5) Only Affordance recall (not consider part node label)
    print("Evaluating affordance only (AFFORDANCE)...")
    for gt_aff in tqdm(gt_aff_instances, desc="Affordance eval"):
        gt_bbox = gt_aff["bbox"]
        gt_label = gt_aff["label"]
        gt_aff_label = gt_aff["aff_label"]

        best_rank = max_k + 1
        # candidates: predicted parts only (only parts have affordance field)
        for pred_part in pred_parts.values():
            score = overlap_fn(gt_bbox, pred_part["bbox"])
            if score <= bbox_th:
                continue

            pred_aff_list = pred_part.get("affordance", [])
            if not isinstance(pred_aff_list, list) or len(pred_aff_list) == 0:
                continue

            best_aff_label_rank = max_k + 1
            for pred_aff in pred_aff_list:
                top_aff_labels = get_top_aff_labels(pred_aff)
                r = rank_of_gt_in_topk(gt_aff_label, top_aff_labels, max_k=max_k)
                if r < best_aff_label_rank:
                    best_aff_label_rank = r

            cand_rank = best_aff_label_rank
            if cand_rank < best_rank:
                best_rank = cand_rank
                if best_rank == 1:
                    break

        for k in ks:
            if best_rank <= k:
                metrics["only AFFORDANCE"][k] += 1

    # ---------------------- Print final summary ---------------------- #
    gt_counts = {
        "Obj": gt_obj_count,
        "Part": gt_part_count,
        "Func": gt_func_count,
        "Spat": gt_spat_count,
        "Affor": gt_aff_count,
        "onlyAffor": gt_aff_count,
    }

    # print_summary_table(gt_counts, metrics, ks)

    if args.debug:
        print(f"[Debug] Effective GT func relations: {gt_func_count - fail_func} (failed: {fail_func})")
        print(f"[Debug] Effective GT spat relations: {gt_spat_count - fail_spat} (failed: {fail_spat})")


    print("===============================================================")
    print(f"For Ctrl+C copy-paste: {args.scene+'/'+args.video}")
    left_cols = ["Obj", "Part", "Func", "Spat", "Affor"]
    
    line1 = ""
    groups1 = ["OBJECT node", "PART node"]
    gt_line = ", ".join([str(gt_counts[c]) for c in left_cols])
    for g in groups1:
        line1 += ", "
        line1 += ", ".join([str(metrics[g][k]) for k in ks])
    print(gt_line + line1)
    
    line2 = ""
    groups2 = [
        "Assoc. Node",
        "FUNCTIONAL triplet",
        "SPATIAL triplet",
        "AFFORDANCE",
        "only AFFORDANCE",
    ]
    for g in groups2:
        line2 += ", "
        line2 += ", ".join([str(metrics[g][k]) for k in ks])
    print(line2)
    print("===============================================================")
    print(" ")

if __name__ == "__main__":
    main()
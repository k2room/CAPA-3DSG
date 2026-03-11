#!/usr/bin/env python3
"""
Evaluate ReplicaSSG predictions produced by CAPA/eval.py-style outputs.

Input per scene
---------------
1) Object file: .pkl.gz
   Typically:
     <scene>/CAPA_1/object/pcd_saves/full_pcd_ram_update.pkl.gz

   This file must contain per-object point clouds. Supported variants:
     - dict with key "objects" -> list[dict] or dict[obj_key -> dict]
     - raw list[dict]
     - raw dict[obj_key -> dict]

   Each object should provide point clouds via one of:
     - pcd_np
     - pcd
     - points
     - point_cloud

2) Scene-graph file: spatial_3d_scene_graph.json
   Typically:
     <scene>/CAPA_1/scene_graph/spatial_3d_scene_graph.json

   Expected structure:
     {
       "object": {
         "obj_0": {"label": "...", "center": [...], "extent": [...]},
         ...
       },
       "spatial_relation": [
         {"pair": ["obj_0", "obj_7"], "label": "near"},
         ...
       ]
     }

Ground-truth files
------------------
This script supports the user's ReplicaSSG layout where GT metadata lives under:

    <dataset_path>/files/
        train_scans.txt / validation_scans.txt / test_scans.txt
        objects.json
        relationships.json
        replica_to_visual_genome.json

GT mesh path is auto-detected from a few common candidates, and can also be
overridden with --gt_mesh_pattern.

Evaluation
----------
- object/node recall@k
- predicate recall@k
- relationship/triplet recall@k

This is a top-k retrieval style evaluation:
- node/object: rank GT object class within the matched predicted node score vector
- predicate: rank GT relation class within the matched predicted edge score vector
- triplet/relationship: rank GT (subject class, object class, relation) within
  the matched edge's joint score tensor

Object matching between GT and predictions:
- default: 3D IoU based matching between GT AABB and predicted AABB
  (threshold controlled by --iou_threshold, default 0.0; internally this means
   positive overlap when threshold == 0 to avoid matching all zero-IoU pairs)
- optional: official ReplicaSSG point-overlap matching retained via
  --match_method point_overlap
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from plyfile import PlyData
except ImportError as e:
    raise ImportError("Missing dependency: plyfile. Install it with `pip install plyfile`.") from e

from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Missing dependency: sentence-transformers. Install it with `pip install sentence-transformers`."
    ) from e

from tqdm import tqdm

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as e:
    raise ImportError(
        "Missing dependency: transformers. Install it with `pip install transformers`."
    ) from e

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------- constants ----------------------------- #

OBJ_CLASS_NAME = "VisualGenome_list"
REL_CLASS_NAME = "VisualGenome_rel"
MAPPING_NAME = "Replica2VisualGenome"
LARGE_RANK = 99999


# ----------------------------- utilities ----------------------------- #

def normalize_text(text: Any) -> str:
    text = str(text)
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace(":", " ")
    text = " ".join(text.split())
    return text.strip().lower()


def softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32) / float(temperature)
    x = x - np.max(x)
    exp_x = np.exp(x)
    denom = np.sum(exp_x)
    if denom <= 0:
        return np.ones_like(exp_x, dtype=np.float32) / max(len(exp_x), 1)
    return (exp_x / denom).astype(np.float32)


def to_plain_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return {}


def as_array3(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size != 3:
        return None
    return arr.astype(np.float32)


def safe_path(p: Optional[str | Path]) -> Optional[Path]:
    if p is None:
        return None
    return Path(p).expanduser().resolve()


def maybe_to_obj_key(value: Any, valid_keys: set[str]) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, (np.integer, int)):
        key = f"obj_{int(value)}"
        if key in valid_keys:
            return key

    value_str = str(value)
    if value_str in valid_keys:
        return value_str

    if value_str.isdigit():
        key = f"obj_{int(value_str)}"
        if key in valid_keys:
            return key

    return None


def extract_point_cloud_array(obj: Dict[str, Any]) -> np.ndarray:
    candidate_keys = ["pcd_np", "pcd", "points", "point_cloud"]
    for key in candidate_keys:
        if key not in obj or obj[key] is None:
            continue

        value = obj[key]
        if hasattr(value, "points"):
            arr = np.asarray(value.points, dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)

        if arr.size == 0:
            continue

        arr = arr.reshape(-1, 3).astype(np.float32)
        if arr.shape[0] > 0:
            return arr

    raise KeyError("Prediction object does not contain a supported point-cloud field.")


def bbox_from_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return np.stack([points.min(axis=0), points.max(axis=0)], axis=0).astype(np.float32)


def bbox_from_center_extent(center: Sequence[float], extent: Sequence[float]) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32).reshape(3)
    extent = np.asarray(extent, dtype=np.float32).reshape(3)
    half = extent / 2.0
    return np.stack([center - half, center + half], axis=0).astype(np.float32)


def bbox_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    a = np.asarray(box_a, dtype=np.float32).reshape(2, 3)
    b = np.asarray(box_b, dtype=np.float32).reshape(2, 3)

    inter_min = np.maximum(a[0], b[0])
    inter_max = np.minimum(a[1], b[1])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_size))

    a_size = np.maximum(a[1] - a[0], 0.0)
    b_size = np.maximum(b[1] - b[0], 0.0)
    a_vol = float(np.prod(a_size))
    b_vol = float(np.prod(b_size))

    union = a_vol + b_vol - inter_vol
    if union <= 0.0:
        return 0.0
    return inter_vol / union


def choose_best_match_per_pred(matrix: np.ndarray, threshold: float, positive_overlap_when_zero: bool = True) -> np.ndarray:
    """
    Keep only the best GT row per predicted column.
    """
    out = np.zeros_like(matrix, dtype=np.float32)
    if matrix.size == 0:
        return out

    num_gt, num_pred = matrix.shape
    for pred_idx in range(num_pred):
        col = matrix[:, pred_idx]
        if col.size == 0:
            continue
        gt_idx = int(np.argmax(col))
        score = float(col[gt_idx])

        if positive_overlap_when_zero and threshold == 0.0:
            valid = score > 0.0
        else:
            valid = score >= threshold

        if valid:
            out[gt_idx, pred_idx] = score
    return out


def recall_at(values: List[int], k: int) -> float:
    if len(values) == 0:
        return 0.0
    return sum(1 for v in values if v < k) / len(values)


def per_class_recall(metric_dict: Dict[int, List[int]], k: int) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for c, values in metric_dict.items():
        if len(values) == 0:
            continue
        result[str(c)] = recall_at(values, k)
    return result


# ----------------------------- dataset layout ----------------------------- #

@dataclass
class DatasetLayout:
    dataset_root: Path
    gt_files_dir: Path


def resolve_dataset_layout(dataset_path: Path) -> DatasetLayout:
    dataset_path = dataset_path.resolve()

    candidates = [
        dataset_path / "files",
        dataset_path / "ReplicaSSG" / "files",
    ]
    for gt_files_dir in candidates:
        if gt_files_dir.exists():
            if gt_files_dir.parent.name == "ReplicaSSG":
                return DatasetLayout(dataset_root=gt_files_dir.parent, gt_files_dir=gt_files_dir)
            return DatasetLayout(dataset_root=dataset_path, gt_files_dir=gt_files_dir)

    raise FileNotFoundError(
        "Could not locate GT metadata directory. Expected one of:\n"
        f"  - {dataset_path / 'files'}\n"
        f"  - {dataset_path / 'ReplicaSSG' / 'files'}"
    )


def resolve_gt_mesh_path(
    layout: DatasetLayout,
    scan_id: str,
    use_aligned_ply: bool = False,
    gt_mesh_pattern: Optional[str] = None,
) -> Path:
    filename = f"labels.instances{'.align' if use_aligned_ply else ''}.annotated.v2.ply"

    if gt_mesh_pattern:
        custom = Path(str(gt_mesh_pattern).format(scan_id=scan_id, filename=filename)).expanduser().resolve()
        if custom.exists():
            return custom
        raise FileNotFoundError(f"Custom GT mesh path does not exist: {custom}")

    roots = [
        layout.dataset_root,
        layout.dataset_root.parent,
    ]

    candidates: List[Path] = []
    for root in roots:
        candidates.extend([
            root / "data" / scan_id / filename,
            root / scan_id / filename,
            root / scan_id / "habitat" / filename,
            root / "ReplicaSSG" / "data" / scan_id / filename,
            root / "ReplicaSSG" / scan_id / filename,
            root / "ReplicaSSG" / scan_id / "habitat" / filename,
        ])

    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        f"Could not locate GT mesh for scan '{scan_id}'. "
        "Use --gt_mesh_pattern to override the auto-detected path."
    )


# ----------------------------- file loading ----------------------------- #

def load_pickle_maybe_gzip(path: Path) -> Any:
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except OSError:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_prediction_objects(result_path: Path) -> List[Dict[str, Any]]:
    data = load_pickle_maybe_gzip(result_path)
    objects = data["objects"] if isinstance(data, dict) and "objects" in data else data

    out: List[Dict[str, Any]] = []

    if isinstance(objects, dict):
        for key, obj in objects.items():
            obj_dict = to_plain_dict(obj)
            if not obj_dict:
                continue
            obj_dict.setdefault("_source_key", str(key))
            out.append(obj_dict)
        return out

    if isinstance(objects, (list, tuple)):
        for idx, obj in enumerate(objects):
            obj_dict = to_plain_dict(obj)
            if not obj_dict:
                continue
            obj_dict.setdefault("_source_index", idx)
            out.append(obj_dict)
        return out

    raise ValueError(
        f"Unsupported object result format in {result_path}. "
        "Expected dict/list or dict with key 'objects'."
    )


def load_scene_graph_json(scene_graph_path: Path) -> Dict[str, Any]:
    with open(scene_graph_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Scene-graph JSON must be a dict: {scene_graph_path}")
    return data


def get_scene_graph_objects(scene_graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = scene_graph.get("object")
    if raw is None:
        raw = scene_graph.get("objects", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def get_scene_graph_relations(scene_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["spatial_relation", "spatial_relations", "relations", "spatial_rel"]:
        value = scene_graph.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def load_results_manifest(manifest_path: Path) -> Dict[str, Dict[str, Optional[Path]]]:
    with open(manifest_path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and ("object" in payload or "spatial_relation" in payload):
        raise ValueError(
            "--results_manifest appears to be a single scene-graph JSON, not a manifest.\n"
            "For a single scene, use --scene/--scan_id with --obj_file and --sg_file."
        )

    base_dir = manifest_path.parent
    resolved: Dict[str, Dict[str, Optional[Path]]] = {}

    def _resolve_path(p: Any) -> Optional[Path]:
        if p is None:
            return None
        p = Path(p)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return p

    if isinstance(payload, dict):
        items = []
        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("scan_id", key)
                items.append(item)
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("results_manifest must be a dict or a list.")

    for item in items:
        if not isinstance(item, dict):
            continue
        scan_id = item.get("scan_id") or item.get("scene") or item.get("scan") or item.get("id")
        if not scan_id:
            raise ValueError("Each manifest entry must contain scan_id / scene / scan / id.")
        obj_file = _resolve_path(item.get("obj_file") or item.get("object_file") or item.get("objects"))
        sg_file = _resolve_path(
            item.get("sg_file")
            or item.get("scene_graph_file")
            or item.get("scene_graph")
            or item.get("edge_file")
            or item.get("edges")
        )
        resolved[str(scan_id)] = {"obj_file": obj_file, "sg_file": sg_file}

    return resolved


# ----------------------------- GT loading ----------------------------- #

@dataclass
class GTBundle:
    scan_ids: List[str]
    class_mapping: Dict[str, Any]
    scan2obj_rel: Dict[str, Dict[str, Any]]
    objid2idx: Dict[str, Dict[int, int]]
    obj_in_rel: Dict[str, set]


def determine_scan_ids(args: argparse.Namespace, layout: DatasetLayout) -> List[str]:
    explicit_scan = args.scene or args.scan_id
    if explicit_scan:
        return [explicit_scan]

    split = args.split
    scan_split = "validation" if split == "val" else split
    split_path = layout.gt_files_dir / f"{scan_split}_scans.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r") as f:
        scan_ids = [line.strip() for line in f.readlines() if line.strip()]
    return scan_ids


def load_replica_gt(layout: DatasetLayout, scan_ids: List[str], skip_no_rel_objects: bool) -> GTBundle:
    obj3d_ann_path = layout.gt_files_dir / "objects.json"
    sg_ann_path = layout.gt_files_dir / "relationships.json"
    class_mapping_path = layout.gt_files_dir / "replica_to_visual_genome.json"

    with open(obj3d_ann_path, "r") as f:
        obj3d_ann = json.load(f)
    with open(sg_ann_path, "r") as f:
        sg_ann = json.load(f)
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)

    scan_ids_set = set(scan_ids)
    scan2obj_rel: Dict[str, Dict[str, Any]] = {}
    objid2idx: Dict[str, Dict[int, int]] = {}
    obj_in_rel: Dict[str, set] = {}

    for scan in sg_ann["scans"]:
        scan_id = scan["scan"]
        if scan_id not in scan_ids_set:
            continue
        obj_in_rel[scan_id] = set()
        for rel in scan["relationships"]:
            s, o, _, _ = rel
            obj_in_rel[scan_id].add(int(s))
            obj_in_rel[scan_id].add(int(o))

    for scan in obj3d_ann["scans"]:
        scan_id = scan["scan"]
        if scan_id not in scan_ids_set:
            continue

        scan2obj_rel[scan_id] = {"obj_id": [], "obj_cls": [], "rel_edge": [], "rel_cls": []}
        objid2idx[scan_id] = {}

        for obj in scan["objects"]:
            obj_id = int(obj["id"])

            if skip_no_rel_objects and obj_id not in obj_in_rel.get(scan_id, set()):
                continue

            raw_label = obj["label"]
            if raw_label not in class_mapping[MAPPING_NAME]:
                continue
            vg_label = class_mapping[MAPPING_NAME][raw_label]
            if vg_label not in class_mapping[OBJ_CLASS_NAME]:
                continue

            cls_idx = class_mapping[OBJ_CLASS_NAME].index(vg_label)
            scan2obj_rel[scan_id]["obj_cls"].append(cls_idx)
            scan2obj_rel[scan_id]["obj_id"].append(obj_id)
            objid2idx[scan_id][obj_id] = len(scan2obj_rel[scan_id]["obj_cls"]) - 1

    for scan in sg_ann["scans"]:
        scan_id = scan["scan"]
        if scan_id not in scan_ids_set:
            continue

        for rel in scan["relationships"]:
            s, o, _, cls = rel
            s = int(s)
            o = int(o)
            if s not in objid2idx[scan_id] or o not in objid2idx[scan_id]:
                continue
            cls_idx = class_mapping[REL_CLASS_NAME].index(cls)
            scan2obj_rel[scan_id]["rel_edge"].append((objid2idx[scan_id][s], objid2idx[scan_id][o]))
            scan2obj_rel[scan_id]["rel_cls"].append([cls_idx])

    return GTBundle(
        scan_ids=scan_ids,
        class_mapping=class_mapping,
        scan2obj_rel=scan2obj_rel,
        objid2idx=objid2idx,
        obj_in_rel=obj_in_rel,
    )


# ----------------------------- label encoders ----------------------------- #

class LabelEncoderBank:
    def __init__(
        self,
        obj_classes: List[str],
        rel_classes: List[str],
        device: str = "cpu",
        batch_size: int = 64,
        temperature: float = 1.0,
    ) -> None:
        self.obj_classes = [normalize_text(x) for x in obj_classes]
        self.rel_classes = [normalize_text(x) for x in rel_classes]
        self.batch_size = batch_size
        self.temperature = temperature

        if device == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = device

        print("Loading CLIP model for object label normalization...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        print("Loading SBERT model for relation label normalization...")
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        print("Encoding ReplicaSSG object vocabulary...")
        self.obj_class_emb = self._encode_clip_texts(self.obj_classes)
        print("Encoding ReplicaSSG relation vocabulary...")
        self.rel_class_emb = self._encode_sbert_texts(self.rel_classes)

        self.obj_score_cache: Dict[str, np.ndarray] = {}
        self.rel_score_cache: Dict[str, np.ndarray] = {}

    def _batched(self, items: List[str]) -> Iterable[List[str]]:
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    def _encode_clip_texts(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 512), dtype=np.float32)

        feats = []
        for batch in tqdm(list(self._batched(texts)), desc="CLIP text", leave=False):
            inputs = self.clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.clip_model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)

    def _encode_sbert_texts(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 384), dtype=np.float32)
        emb = self.sbert.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def precompute(self, obj_labels: Iterable[str], rel_labels: Iterable[str]) -> None:
        obj_labels = sorted({normalize_text(x) for x in obj_labels if str(x).strip()})
        rel_labels = sorted({normalize_text(x) for x in rel_labels if str(x).strip()})

        missing_obj = [x for x in obj_labels if x not in self.obj_score_cache]
        missing_rel = [x for x in rel_labels if x not in self.rel_score_cache]

        if missing_obj:
            print(f"Precomputing {len(missing_obj)} unique object labels...")
            pred_emb = self._encode_clip_texts(missing_obj)
            sims = pred_emb @ self.obj_class_emb.T
            for label, sim in zip(missing_obj, sims):
                self.obj_score_cache[label] = softmax_np(sim, temperature=self.temperature)

        if missing_rel:
            print(f"Precomputing {len(missing_rel)} unique relation labels...")
            pred_emb = self._encode_sbert_texts(missing_rel)
            sims = pred_emb @ self.rel_class_emb.T
            for label, sim in zip(missing_rel, sims):
                self.rel_score_cache[label] = softmax_np(sim, temperature=self.temperature)

    def object_scores(self, label: str) -> np.ndarray:
        label = normalize_text(label)
        if label not in self.obj_score_cache:
            pred_emb = self._encode_clip_texts([label])[0]
            sim = pred_emb @ self.obj_class_emb.T
            self.obj_score_cache[label] = softmax_np(sim, temperature=self.temperature)
        return self.obj_score_cache[label]

    def relation_scores(self, label: str) -> np.ndarray:
        label = normalize_text(label)
        if label not in self.rel_score_cache:
            pred_emb = self._encode_sbert_texts([label])[0]
            sim = pred_emb @ self.rel_class_emb.T
            self.rel_score_cache[label] = softmax_np(sim, temperature=self.temperature)
        return self.rel_score_cache[label]


# ----------------------------- prediction conversion ----------------------------- #

def get_prediction_fallback_label(pred_obj: Dict[str, Any]) -> str:
    for key in ["refined_obj_tag", "label", "class_name", "name"]:
        if key in pred_obj and pred_obj[key] is not None:
            value = pred_obj[key]
            if isinstance(value, str):
                return normalize_text(value)
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                return normalize_text(value[0])
    return "unknown"


def get_scene_graph_object_bbox(obj_info: Dict[str, Any]) -> Optional[np.ndarray]:
    center = as_array3(obj_info.get("center"))
    extent = as_array3(obj_info.get("extent"))
    if center is None or extent is None:
        return None
    return bbox_from_center_extent(center, extent)


def infer_explicit_obj_key(pred_obj: Dict[str, Any], valid_keys: set[str]) -> Optional[str]:
    for field in ["_source_key", "obj_key", "key", "id", "obj_id", "object_id", "instance_id", "name"]:
        key = maybe_to_obj_key(pred_obj.get(field), valid_keys)
        if key is not None:
            return key
    return None


def build_pred_entries(
    objects: List[Dict[str, Any]],
    scene_graph_objects: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    valid_keys = set(scene_graph_objects.keys())
    entries: List[Dict[str, Any]] = []

    for idx, obj in enumerate(objects):
        try:
            pcd = extract_point_cloud_array(obj)
        except Exception:
            continue
        if pcd.size == 0 or pcd.shape[0] == 0:
            continue

        pred_bbox = None
        if as_array3(obj.get("center")) is not None and as_array3(obj.get("extent")) is not None:
            pred_bbox = bbox_from_center_extent(obj["center"], obj["extent"])
        else:
            pred_bbox = bbox_from_points(pcd)

        entry = {
            "source_index": idx,
            "obj": obj,
            "pcd": pcd.astype(np.float32),
            "center": pcd.mean(axis=0).astype(np.float32),
            "bbox": pred_bbox.astype(np.float32),
            "explicit_key": infer_explicit_obj_key(obj, valid_keys),
        }
        entries.append(entry)

    return entries


def map_pred_entries_to_scene_graph(
    pred_entries: List[Dict[str, Any]],
    scene_graph_objects: Dict[str, Dict[str, Any]],
) -> Dict[int, str]:
    """
    Returns mapping: pred_entry_index -> scene_graph_obj_key
    """
    mapping: Dict[int, str] = {}
    used_keys: set[str] = set()
    valid_keys = set(scene_graph_objects.keys())

    # 1) explicit id/key based matching
    unresolved: List[int] = []
    for i, entry in enumerate(pred_entries):
        key = entry.get("explicit_key")
        if key is not None and key in valid_keys and key not in used_keys:
            mapping[i] = key
            used_keys.add(key)
        else:
            unresolved.append(i)

    # 2) straightforward obj_i matching if the scene-graph has that convention
    if unresolved:
        for i in list(unresolved):
            enum_key = f"obj_{pred_entries[i]['source_index']}"
            if enum_key in valid_keys and enum_key not in used_keys:
                mapping[i] = enum_key
                used_keys.add(enum_key)
        unresolved = [i for i in unresolved if i not in mapping]

    # 3) center matching via Hungarian assignment
    remaining_keys = [k for k in scene_graph_objects.keys() if k not in used_keys]
    if unresolved and remaining_keys:
        pred_centers: List[np.ndarray] = []
        valid_pred_indices: List[int] = []
        sg_centers: List[np.ndarray] = []
        valid_sg_keys: List[str] = []

        for i in unresolved:
            pred_centers.append(np.asarray(pred_entries[i]["center"], dtype=np.float32))
            valid_pred_indices.append(i)

        for key in remaining_keys:
            center = as_array3(scene_graph_objects[key].get("center"))
            if center is None:
                bbox = get_scene_graph_object_bbox(scene_graph_objects[key])
                if bbox is not None:
                    center = (bbox[0] + bbox[1]) / 2.0
            if center is None:
                continue
            sg_centers.append(center.astype(np.float32))
            valid_sg_keys.append(key)

        if pred_centers and sg_centers:
            pred_mat = np.stack(pred_centers, axis=0)
            sg_mat = np.stack(sg_centers, axis=0)

            # pairwise L2 distances
            dists = np.linalg.norm(pred_mat[:, None, :] - sg_mat[None, :, :], axis=-1)
            row_idx, col_idx = linear_sum_assignment(dists)
            for r, c in zip(row_idx.tolist(), col_idx.tolist()):
                pred_i = valid_pred_indices[r]
                key = valid_sg_keys[c]
                if key not in used_keys:
                    mapping[pred_i] = key
                    used_keys.add(key)

    return mapping


def convert_scan_result_to_prediction_from_scene_graph(
    objects: List[Dict[str, Any]],
    scene_graph: Dict[str, Any],
    encoder_bank: LabelEncoderBank,
    num_obj_classes: int,
    num_rel_classes: int,
) -> Dict[str, Any]:
    sg_objects = get_scene_graph_objects(scene_graph)
    sg_relations = get_scene_graph_relations(scene_graph)

    pred_entries = build_pred_entries(objects, sg_objects)
    entry_to_objkey = map_pred_entries_to_scene_graph(pred_entries, sg_objects)

    pred_pcds: List[np.ndarray] = []
    pred_bboxes: List[np.ndarray] = []
    pred_obj_scores: List[np.ndarray] = []
    objkey_to_new_idx: Dict[str, int] = {}

    for entry_idx, entry in enumerate(pred_entries):
        obj_key = entry_to_objkey.get(entry_idx)
        if obj_key is None:
            continue

        obj_info = sg_objects.get(obj_key, {})
        label = normalize_text(obj_info.get("label", get_prediction_fallback_label(entry["obj"])))
        scores = encoder_bank.object_scores(label)

        pred_pcds.append(entry["pcd"])
        pred_obj_scores.append(scores.astype(np.float32))

        json_bbox = get_scene_graph_object_bbox(obj_info)
        pred_bboxes.append((json_bbox if json_bbox is not None else entry["bbox"]).astype(np.float32))

        objkey_to_new_idx[obj_key] = len(pred_pcds) - 1

    pair_to_rel_scores: Dict[Tuple[int, int], np.ndarray] = {}
    for rel in sg_relations:
        pair = rel.get("pair")
        label = rel.get("label")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or label is None:
            continue

        sub_key = str(pair[0])
        obj_key = str(pair[1])
        if sub_key not in objkey_to_new_idx or obj_key not in objkey_to_new_idx:
            continue

        sub_new = objkey_to_new_idx[sub_key]
        obj_new = objkey_to_new_idx[obj_key]
        if sub_new == obj_new:
            continue

        rel_scores = encoder_bank.relation_scores(normalize_text(label)).astype(np.float32)
        pair_idx = (sub_new, obj_new)
        if pair_idx in pair_to_rel_scores:
            pair_to_rel_scores[pair_idx] = np.maximum(pair_to_rel_scores[pair_idx], rel_scores)
        else:
            pair_to_rel_scores[pair_idx] = rel_scores

    if pred_obj_scores:
        cls = np.stack(pred_obj_scores, axis=0)
    else:
        cls = np.zeros((0, num_obj_classes), dtype=np.float32)

    if pair_to_rel_scores:
        edge_pairs = list(pair_to_rel_scores.keys())
        edge_cls = np.stack([pair_to_rel_scores[pair] for pair in edge_pairs], axis=0)
        edge_index = np.asarray(edge_pairs, dtype=np.int64).T
    else:
        edge_cls = np.zeros((0, num_rel_classes), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return {
        "cls": cls,
        "pcd": pred_pcds,
        "bbox": pred_bboxes,
        "edge_cls": edge_cls,
        "edge_index": edge_index,
    }


# ----------------------------- prediction path resolution ----------------------------- #

def determine_result_specs(
    args: argparse.Namespace,
    scan_ids: List[str],
) -> Dict[str, Dict[str, Optional[Path]]]:
    if args.results_manifest is not None:
        specs = load_results_manifest(Path(args.results_manifest))
        return {scan_id: specs.get(scan_id, {"obj_file": None, "sg_file": None}) for scan_id in scan_ids}

    single_sg = args.sg_file or args.scene_graph_file or args.edge_file
    if len(scan_ids) == 1 and args.obj_file and single_sg:
        return {
            scan_ids[0]: {
                "obj_file": safe_path(args.obj_file),
                "sg_file": safe_path(single_sg),
            }
        }

    sg_pattern = args.sg_pattern or args.scene_graph_pattern or args.edge_pattern
    if args.obj_pattern and sg_pattern:
        specs: Dict[str, Dict[str, Optional[Path]]] = {}
        for scan_id in scan_ids:
            specs[scan_id] = {
                "obj_file": safe_path(str(args.obj_pattern).format(scan_id=scan_id)),
                "sg_file": safe_path(str(sg_pattern).format(scan_id=scan_id)),
            }
        return specs

    raise ValueError(
        "Could not determine prediction files. Provide either:\n"
        "  (1) --results_manifest\n"
        "  (2) --scene/--scan_id with --obj_file and --sg_file\n"
        "  (3) --obj_pattern and --sg_pattern containing {scan_id}"
    )


# ----------------------------- GT mesh processing ----------------------------- #

def load_gt_object_points_and_bboxes(
    mesh_path: Path,
    objid2idx_for_scan: Dict[int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    ply = PlyData.read(str(mesh_path))
    points = np.stack(
        [ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]],
        axis=1,
    ).astype(np.float32)
    object_ids = np.asarray(ply["vertex"]["objectId"], dtype=np.int64)

    gt_points_list: List[Optional[np.ndarray]] = [None] * len(objid2idx_for_scan)
    gt_bbox_list: List[Optional[np.ndarray]] = [None] * len(objid2idx_for_scan)

    for obj_id, gt_idx in objid2idx_for_scan.items():
        mask = object_ids == int(obj_id)
        obj_points = points[mask]
        if obj_points.shape[0] == 0:
            gt_points_list[gt_idx] = np.zeros((0, 3), dtype=np.float32)
            gt_bbox_list[gt_idx] = np.zeros((2, 3), dtype=np.float32)
        else:
            gt_points_list[gt_idx] = obj_points.astype(np.float32)
            gt_bbox_list[gt_idx] = bbox_from_points(obj_points)

    return [x if x is not None else np.zeros((0, 3), dtype=np.float32) for x in gt_points_list], [
        x if x is not None else np.zeros((2, 3), dtype=np.float32) for x in gt_bbox_list
    ]


# ----------------------------- matching ----------------------------- #

def match_gt_pred_by_iou(
    gt_bboxes: List[np.ndarray],
    pred_bboxes: List[np.ndarray],
    iou_threshold: float,
) -> np.ndarray:
    gt2pred = -np.ones((2, len(gt_bboxes)), dtype=np.int64)
    gt2pred[0] = np.arange(len(gt_bboxes), dtype=np.int64)

    if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        return gt2pred

    iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float32)
    for gt_idx, gt_box in enumerate(gt_bboxes):
        for pred_idx, pred_box in enumerate(pred_bboxes):
            iou_matrix[gt_idx, pred_idx] = bbox_iou_3d(gt_box, pred_box)

    filtered = choose_best_match_per_pred(iou_matrix, threshold=iou_threshold, positive_overlap_when_zero=True)

    for gt_idx in range(filtered.shape[0]):
        best_pred = int(np.argmax(filtered[gt_idx]))
        if filtered[gt_idx, best_pred] > 0.0 or (iou_threshold > 0 and filtered[gt_idx, best_pred] >= iou_threshold):
            gt2pred[1, gt_idx] = best_pred

    return gt2pred

def compute_gt_pred_iou_matrix(
    gt_bboxes: List[np.ndarray],
    pred_bboxes: List[np.ndarray],
) -> np.ndarray:
    iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float32)
    if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        return iou_matrix

    for gt_idx, gt_box in enumerate(gt_bboxes):
        for pred_idx, pred_box in enumerate(pred_bboxes):
            iou_matrix[gt_idx, pred_idx] = bbox_iou_3d(gt_box, pred_box)
    return iou_matrix


def match_gt_pred_by_point_overlap(
    gt_points_list: List[np.ndarray],
    pred_pcds: List[np.ndarray],
    eval_overlap_threshold: float,
) -> np.ndarray:
    gt2pred = -np.ones((2, len(gt_points_list)), dtype=np.int64)
    gt2pred[0] = np.arange(len(gt_points_list), dtype=np.int64)

    if len(gt_points_list) == 0 or len(pred_pcds) == 0:
        return gt2pred

    # flatten GT points once
    gt_points_all = []
    gt_point_gt_idx = []
    for gt_idx, pts in enumerate(gt_points_list):
        if pts.shape[0] == 0:
            continue
        gt_points_all.append(pts)
        gt_point_gt_idx.append(np.full((pts.shape[0],), gt_idx, dtype=np.int64))

    if len(gt_points_all) == 0:
        return gt2pred

    gt_points_all_arr = np.concatenate(gt_points_all, axis=0)
    gt_point_gt_idx_arr = np.concatenate(gt_point_gt_idx, axis=0)
    gt_kdtree = KDTree(gt_points_all_arr)

    overlap_count = np.zeros((len(gt_points_list), len(pred_pcds)), dtype=np.float32)

    for pred_idx, seg in enumerate(pred_pcds):
        seg = np.asarray(seg, dtype=np.float32).reshape(-1, 3)
        pred_num_points = len(seg)
        if pred_num_points == 0:
            continue

        _, indices = gt_kdtree.query(seg, distance_upper_bound=eval_overlap_threshold)
        valid = indices != gt_kdtree.n
        matched_gt_idx = gt_point_gt_idx_arr[indices[valid]]

        for gt_idx in range(len(gt_points_list)):
            overlap_count[gt_idx, pred_idx] = np.count_nonzero(matched_gt_idx == gt_idx)

        overlap_percentage = overlap_count[:, pred_idx] / float(pred_num_points)
        sorted_gt_idx = np.flip(np.argsort(overlap_count[:, pred_idx], kind="stable"))
        if len(sorted_gt_idx) == 0:
            continue

        max_gt_idx = sorted_gt_idx[0]
        second_ratio = 0.0
        if len(sorted_gt_idx) > 1 and overlap_percentage[max_gt_idx] > 0:
            second_gt_idx = sorted_gt_idx[1]
            second_ratio = overlap_percentage[second_gt_idx] / overlap_percentage[max_gt_idx]

        if overlap_percentage[max_gt_idx] < 0.5 or second_ratio > 0.75:
            overlap_count[:, pred_idx] = 0
        else:
            overlap_count[np.arange(len(gt_points_list)) != max_gt_idx, pred_idx] = 0

    for gt_idx in range(len(gt_points_list)):
        max_pred_idx = int(np.argmax(overlap_count[gt_idx]))
        if overlap_count[gt_idx, max_pred_idx] == 0:
            continue
        gt2pred[1, gt_idx] = max_pred_idx

    return gt2pred


# ----------------------------- evaluation core ----------------------------- #

def evaluate_replica_predictions(
    layout: DatasetLayout,
    gt_bundle: GTBundle,
    predictions: Dict[str, Dict[str, Any]],
    match_method: str,
    iou_threshold: float,
    eval_overlap_threshold: float,
    ks: List[int],
    use_aligned_ply: bool = False,
    gt_mesh_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    class_mapping = gt_bundle.class_mapping
    scan2obj_rel = gt_bundle.scan2obj_rel
    objid2idx = gt_bundle.objid2idx

    topk = {
        "object": [],
        "predicate": [],
        "relationship": [],
        "object_per_class": {c: [] for c in range(len(class_mapping[OBJ_CLASS_NAME]))},
        "predicate_per_class": {c: [] for c in range(len(class_mapping[REL_CLASS_NAME]))},
    }
    matched_counts = {
        "object": 0,
        "predicate": 0,
        "relationship": 0,
    }

    for scan_id in tqdm(gt_bundle.scan_ids, desc="ReplicaSSG evaluation"):
        if scan_id not in scan2obj_rel:
            continue

        node_gt = np.array(scan2obj_rel[scan_id]["obj_cls"], dtype=np.int64)
        edge_gt = np.array(scan2obj_rel[scan_id]["rel_cls"], dtype=np.int64)
        edge_index_gt = np.array(scan2obj_rel[scan_id]["rel_edge"], dtype=np.int64)

        gt_mesh_path = resolve_gt_mesh_path(
            layout=layout,
            scan_id=scan_id,
            use_aligned_ply=use_aligned_ply,
            gt_mesh_pattern=gt_mesh_pattern,
        )
        gt_points_list, gt_bbox_list = load_gt_object_points_and_bboxes(gt_mesh_path, objid2idx[scan_id])

        pred = predictions.get(scan_id)
        if pred is None:
            pred = {
                "cls": np.zeros((0, len(class_mapping[OBJ_CLASS_NAME])), dtype=np.float32),
                "pcd": [],
                "bbox": [],
                "edge_cls": np.zeros((0, len(class_mapping[REL_CLASS_NAME])), dtype=np.float32),
                "edge_index": np.zeros((2, 0), dtype=np.int64),
            }

        pred_bboxes = [np.asarray(x, dtype=np.float32).reshape(2, 3) for x in pred.get("bbox", [])]

        if match_method == "iou":
            gt2pred = match_gt_pred_by_iou(
                gt_bboxes=gt_bbox_list,
                pred_bboxes=[np.asarray(x, dtype=np.float32).reshape(2, 3) for x in pred.get("bbox", [])],
                iou_threshold=iou_threshold,
            )
        elif match_method == "point_overlap":
            gt2pred = match_gt_pred_by_point_overlap(
                gt_points_list=gt_points_list,
                pred_pcds=[np.asarray(x, dtype=np.float32).reshape(-1, 3) for x in pred.get("pcd", [])],
                eval_overlap_threshold=eval_overlap_threshold,
            )
        else:
            raise ValueError(f"Unsupported match_method: {match_method}")

        iou_matrix = compute_gt_pred_iou_matrix(gt_bbox_list, pred_bboxes)

        node_pred = np.asarray(pred["cls"], dtype=np.float32)
        edge_pred = np.asarray(pred["edge_cls"], dtype=np.float32)
        edge_index_pred = np.asarray(pred["edge_index"], dtype=np.int64)
        edge_index_pred_list = edge_index_pred.transpose().tolist() if edge_index_pred.size > 0 else []
        gt2pred_map = {int(gt2pred[0, i]): int(gt2pred[1, i]) for i in range(gt2pred.shape[1])}

        # # Node / Object Recall
        # for i in range(len(node_gt)):
        #     gt_idx = int(gt2pred[0, i])
        #     pred_idx = int(gt2pred[1, i])
        #     gt_cls = int(node_gt[gt_idx])

        #     if pred_idx < 0 or pred_idx >= len(node_pred):
        #         topk["object"].append(LARGE_RANK)
        #         topk["object_per_class"][gt_cls].append(LARGE_RANK)
        #         continue

        #     matched_counts["object"] += 1

        #     pred_scores = node_pred[pred_idx]
        #     sorted_args = np.flip(np.argsort(pred_scores, kind="stable"))
        #     rank = int(np.nonzero(sorted_args == gt_cls)[0].item())
        #     topk["object"].append(rank)
        #     topk["object_per_class"][gt_cls].append(rank)
        # Node / Object Recall (IoU>0 candidate search)
        for gt_idx in range(len(node_gt)):
            gt_cls = int(node_gt[gt_idx])
            candidates = np.where(iou_matrix[gt_idx] > 0.0)[0].tolist()

            best_rank = LARGE_RANK
            for pred_idx in candidates:
                if pred_idx >= len(node_pred):
                    continue
                pred_scores = node_pred[pred_idx]
                sorted_args = np.flip(np.argsort(pred_scores, kind="stable"))
                rank = int(np.nonzero(sorted_args == gt_cls)[0].item())
                best_rank = min(best_rank, rank)

            if best_rank < LARGE_RANK:
                matched_counts["object"] += 1

            topk["object"].append(best_rank)
            topk["object_per_class"][gt_cls].append(best_rank)

        # # Predicate Recall
        # for i in range(len(edge_gt)):
        #     gt_rel = int(edge_gt[i][0])
        #     sub_idx_gt = int(edge_index_gt[i, 0])
        #     obj_idx_gt = int(edge_index_gt[i, 1])

        #     sub_idx_pred = gt2pred_map.get(sub_idx_gt, -1)
        #     obj_idx_pred = gt2pred_map.get(obj_idx_gt, -1)

        #     if [sub_idx_pred, obj_idx_pred] not in edge_index_pred_list:
        #         topk["predicate"].append(LARGE_RANK)
        #         topk["predicate_per_class"][gt_rel].append(LARGE_RANK)
        #         continue

        #     matched_counts["predicate"] += 1

        #     pred_idx = edge_index_pred_list.index([sub_idx_pred, obj_idx_pred])
        #     pred_rel = edge_pred[pred_idx]
        #     sorted_args = np.flip(np.argsort(pred_rel, kind="stable"))
        #     rank = int(np.nonzero(sorted_args == gt_rel)[0].item())
        #     topk["predicate"].append(rank)
        #     topk["predicate_per_class"][gt_rel].append(rank)
        
        # Associated Node Pair Recall (IoU>0 candidate search)
        for i in range(len(edge_gt)):
            gt_rel = int(edge_gt[i][0])  # bucket용으로만 유지
            sub_idx_gt = int(edge_index_gt[i, 0])
            obj_idx_gt = int(edge_index_gt[i, 1])

            gt_sub_cls = int(node_gt[sub_idx_gt])
            gt_obj_cls = int(node_gt[obj_idx_gt])

            subj_candidates = np.where(iou_matrix[sub_idx_gt] > 0.0)[0].tolist()
            obj_candidates = np.where(iou_matrix[obj_idx_gt] > 0.0)[0].tolist()

            best_pair_rank = LARGE_RANK

            for sub_idx_pred in subj_candidates:
                for obj_idx_pred in obj_candidates:
                    if sub_idx_pred == obj_idx_pred:
                        continue
                    if sub_idx_pred >= len(node_pred) or obj_idx_pred >= len(node_pred):
                        continue

                    if [sub_idx_pred, obj_idx_pred] not in edge_index_pred_list and \
                    [obj_idx_pred, sub_idx_pred] not in edge_index_pred_list:
                        continue

                    pred_sub_scores = node_pred[sub_idx_pred]
                    pred_obj_scores = node_pred[obj_idx_pred]

                    sub_sorted = np.flip(np.argsort(pred_sub_scores, kind="stable"))
                    obj_sorted = np.flip(np.argsort(pred_obj_scores, kind="stable"))

                    sub_rank = int(np.nonzero(sub_sorted == gt_sub_cls)[0].item())
                    obj_rank = int(np.nonzero(obj_sorted == gt_obj_cls)[0].item())

                    pair_rank = max(sub_rank, obj_rank)
                    best_pair_rank = min(best_pair_rank, pair_rank)

            if best_pair_rank < LARGE_RANK:
                matched_counts["predicate"] += 1

            topk["predicate"].append(best_pair_rank)
            topk["predicate_per_class"][gt_rel].append(best_pair_rank)

        # # Triplet / Relationship Recall
        # for i in range(len(edge_gt)):
        #     gt_rel = int(edge_gt[i][0])
        #     sub_idx_gt = int(edge_index_gt[i, 0])
        #     obj_idx_gt = int(edge_index_gt[i, 1])

        #     sub_idx_pred = gt2pred_map.get(sub_idx_gt, -1)
        #     obj_idx_pred = gt2pred_map.get(obj_idx_gt, -1)

        #     if [sub_idx_pred, obj_idx_pred] not in edge_index_pred_list:
        #         topk["relationship"].append(LARGE_RANK)
        #         continue
        #     if sub_idx_pred < 0 or obj_idx_pred < 0:
        #         topk["relationship"].append(LARGE_RANK)
        #         continue
        #     if sub_idx_pred >= len(node_pred) or obj_idx_pred >= len(node_pred):
        #         topk["relationship"].append(LARGE_RANK)
        #         continue
            
        #     matched_counts["relationship"] += 1

        #     pred_idx = edge_index_pred_list.index([sub_idx_pred, obj_idx_pred])
        #     pred_rel = edge_pred[pred_idx]
        #     pred_sub = node_pred[sub_idx_pred]
        #     pred_obj = node_pred[obj_idx_pred]

        #     gt_sub = int(node_gt[sub_idx_gt])
        #     gt_obj = int(node_gt[obj_idx_gt])

        #     so_preds = np.einsum("n,m->nm", pred_sub, pred_obj)
        #     conf_matrix = np.einsum("nm,k->nmk", so_preds, pred_rel)
        #     _, cls_n, rel_k = conf_matrix.shape
        #     flat_conf = conf_matrix.flatten()
        #     sorted_args = np.flip(np.argsort(flat_conf, kind="stable"))

        #     gt_index = (gt_sub * cls_n + gt_obj) * rel_k + gt_rel
        #     rank = int(np.nonzero(sorted_args == gt_index)[0].item())
        #     topk["relationship"].append(rank)

        # Triplet / Relationship Recall (IoU>0 candidate pair search)
        for i in range(len(edge_gt)):
            gt_rel = int(edge_gt[i][0])
            sub_idx_gt = int(edge_index_gt[i, 0])
            obj_idx_gt = int(edge_index_gt[i, 1])

            gt_sub = int(node_gt[sub_idx_gt])
            gt_obj = int(node_gt[obj_idx_gt])

            subj_candidates = np.where(iou_matrix[sub_idx_gt] > 0.0)[0].tolist()
            obj_candidates = np.where(iou_matrix[obj_idx_gt] > 0.0)[0].tolist()

            best_triplet_rank = LARGE_RANK

            for sub_idx_pred in subj_candidates:
                for obj_idx_pred in obj_candidates:
                    if sub_idx_pred == obj_idx_pred:
                        continue
                    if sub_idx_pred >= len(node_pred) or obj_idx_pred >= len(node_pred):
                        continue

                    edge_found = False
                    pred_idx = -1
                    if [sub_idx_pred, obj_idx_pred] in edge_index_pred_list:
                        pred_idx = edge_index_pred_list.index([sub_idx_pred, obj_idx_pred])
                        edge_found = True
                    elif [obj_idx_pred, sub_idx_pred] in edge_index_pred_list:
                        pred_idx = edge_index_pred_list.index([obj_idx_pred, sub_idx_pred])
                        edge_found = True

                    if not edge_found:
                        continue

                    pred_rel = edge_pred[pred_idx]
                    pred_sub = node_pred[sub_idx_pred]
                    pred_obj = node_pred[obj_idx_pred]

                    so_preds = np.einsum("n,m->nm", pred_sub, pred_obj)
                    conf_matrix = np.einsum("nm,k->nmk", so_preds, pred_rel)
                    cls_n = conf_matrix.shape[1]
                    rel_k = conf_matrix.shape[2]
                    flat_conf = conf_matrix.flatten()
                    sorted_args = np.flip(np.argsort(flat_conf, kind="stable"))

                    gt_index = (gt_sub * cls_n + gt_obj) * rel_k + gt_rel
                    rank = int(np.nonzero(sorted_args == gt_index)[0].item())
                    best_triplet_rank = min(best_triplet_rank, rank)

            if best_triplet_rank < LARGE_RANK:
                matched_counts["relationship"] += 1

            topk["relationship"].append(best_triplet_rank)

    summary = {
        "ks": ks,
        "matching": {
            "method": match_method,
            "iou_threshold": float(iou_threshold),
            "point_overlap_radius": float(eval_overlap_threshold),
        },
        "object": {f"R@{k}": recall_at(topk["object"], k) for k in ks},
        "predicate": {f"R@{k}": recall_at(topk["predicate"], k) for k in ks},
        "relationship": {f"R@{k}": recall_at(topk["relationship"], k) for k in ks},
        "object_per_class": {f"R@{k}": per_class_recall(topk["object_per_class"], k) for k in ks},
        "predicate_per_class": {f"R@{k}": per_class_recall(topk["predicate_per_class"], k) for k in ks},
        "object_class_names": class_mapping[OBJ_CLASS_NAME],
        "predicate_class_names": class_mapping[REL_CLASS_NAME],
        # "counts": {
        #     "num_object_gt": len(topk["object"]),
        #     "num_predicate_gt": len(topk["predicate"]),
        #     "num_relationship_gt": len(topk["relationship"]),
        # },
        "counts": {
            "num_object_gt": len(topk["object"]),
            "num_predicate_gt": len(topk["predicate"]),
            "num_relationship_gt": len(topk["relationship"]),
            "matched_object_gt": matched_counts["object"],
            "matched_predicate_gt": matched_counts["predicate"],
            "matched_relationship_gt": matched_counts["relationship"],
        },
        "counts_at_k": {
            f"R@{k}": {
                "correct_object": sum(1 for v in topk["object"] if v < k),
                "correct_predicate": sum(1 for v in topk["predicate"] if v < k),
                "correct_relationship": sum(1 for v in topk["relationship"] if v < k),
            }
            for k in ks
        },
    }

    summary["object_macro"] = {}
    summary["predicate_macro"] = {}
    for k in ks:
        obj_values = list(summary["object_per_class"][f"R@{k}"].values())
        pred_values = list(summary["predicate_per_class"][f"R@{k}"].values())
        summary["object_macro"][f"R@{k}"] = float(np.mean(obj_values)) if obj_values else 0.0
        summary["predicate_macro"][f"R@{k}"] = float(np.mean(pred_values)) if pred_values else 0.0

    # eval.py-friendly aliases
    summary["node"] = copy.deepcopy(summary["object"])
    summary["triplet"] = copy.deepcopy(summary["relationship"])
    summary["node_per_class"] = copy.deepcopy(summary["object_per_class"])
    summary["node_macro"] = copy.deepcopy(summary["object_macro"])

    return summary


# ----------------------------- formatting ----------------------------- #

def format_summary_text(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    ks = summary["ks"]

    # lines.append("ReplicaSSG evaluation summary")
    # lines.append(f"Matching: {summary['matching']}")
    # lines.append(f"GT counts: {summary['counts']}")
    # lines.append("")
    lines.append("ReplicaSSG evaluation summary")
    lines.append(f"Matching: {summary['matching']}")
    lines.append("Base counts:")
    lines.append(f"  Object GT: {summary['counts']['num_object_gt']}")
    lines.append(f"  Object matched: {summary['counts']['matched_object_gt']}")
    lines.append(f"  Predicate GT: {summary['counts']['num_predicate_gt']}")
    lines.append(f"  Predicate matched: {summary['counts']['matched_predicate_gt']}")
    lines.append(f"  Relationship GT: {summary['counts']['num_relationship_gt']}")
    lines.append(f"  Relationship matched: {summary['counts']['matched_relationship_gt']}")
    lines.append("")

    lines.append("Counts at each k:")
    for k in ks:
        key = f"R@{k}"
        lines.append(f"  {key}:")
        lines.append(f"    Correct object: {summary['counts_at_k'][key]['correct_object']} / {summary['counts']['num_object_gt']}")
        lines.append(f"    Correct predicate: {summary['counts_at_k'][key]['correct_predicate']} / {summary['counts']['num_predicate_gt']}")
        lines.append(f"    Correct relationship: {summary['counts_at_k'][key]['correct_relationship']} / {summary['counts']['num_relationship_gt']}")
    lines.append("")

    lines.append("Node/Object:")
    for k in ks:
        lines.append(f"  R@{k}: {summary['object'][f'R@{k}']:.6f}")
    lines.append("")

    lines.append("Predicate:")
    for k in ks:
        lines.append(f"  R@{k}: {summary['predicate'][f'R@{k}']:.6f}")
    lines.append("")

    lines.append("Triplet/Relationship:")
    for k in ks:
        lines.append(f"  R@{k}: {summary['relationship'][f'R@{k}']:.6f}")
    lines.append("")

    lines.append("Object per class:")
    obj_names = summary["object_class_names"]
    obj_by_first_k = summary["object_per_class"][f"R@{ks[0]}"]
    for idx_str, value in obj_by_first_k.items():
        idx = int(idx_str)
        lines.append(f"  [{idx:02d}] {obj_names[idx]}: {value:.6f}")
    lines.append(f"  Macro avg: {summary['object_macro'][f'R@{ks[0]}']:.6f}")
    lines.append("")

    lines.append("Predicate per class:")
    rel_names = summary["predicate_class_names"]
    rel_by_first_k = summary["predicate_per_class"][f"R@{ks[0]}"]
    for idx_str, value in rel_by_first_k.items():
        idx = int(idx_str)
        lines.append(f"  [{idx:02d}] {rel_names[idx]}: {value:.6f}")
    lines.append(f"  Macro avg: {summary['predicate_macro'][f'R@{ks[0]}']:.6f}")

    return "\n".join(lines)


def format_summary_text_simple(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    ks = summary["ks"]
    lines.append(f"{summary['counts']['num_object_gt']} {summary['counts']['num_predicate_gt']} ")
    temp_obj = []
    temp_pred = []
    temp_trip = []

    for k in ks:
        key = f"R@{k}"
        temp_obj.append(summary['counts_at_k'][key]['correct_object'])
        temp_pred.append(summary['counts_at_k'][key]['correct_predicate'])
        temp_trip.append(summary['counts_at_k'][key]['correct_relationship'])
    
    for i in range(len(ks)):
        print(temp_obj[i], end=' ')
    for i in range(len(ks)):
        print(temp_pred[i], end=' ')
    for i in range(len(ks)):
        print(temp_trip[i], end=' ')
    print(" ")

    return "\n".join(lines)


# ----------------------------- CLI ----------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # dataset root
    parser.add_argument("--dataset", type=str, default="ReplicaSSG", help="Accepted for eval.py-style compatibility.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="ReplicaSSG dataset root. Supports either <dataset_path>/files/... or <dataset_path>/ReplicaSSG/files/...",
    )
    parser.add_argument("--root_path", type=str, default=None, help="Alias of --dataset_path.")

    # scan selection
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--scan_id", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None, help="Alias of --scan_id.")
    parser.add_argument("--video", type=str, default=None, help="Ignored. Kept only for CLI compatibility.")

    # prediction inputs
    parser.add_argument("--obj_file", type=str, default=None, help="Single-scan object .pkl.gz file.")
    parser.add_argument("--sg_file", type=str, default=None, help="Single-scan spatial_3d_scene_graph.json file.")
    parser.add_argument("--scene_graph_file", type=str, default=None, help="Alias of --sg_file.")
    parser.add_argument("--edge_file", type=str, default=None, help="Alias of --sg_file for backward compatibility.")
    parser.add_argument("--part_file", type=str, default=None, help="Ignored. Kept only for CLI compatibility.")

    parser.add_argument("--results_manifest", type=str, default=None, help="JSON manifest for multiple scans.")
    parser.add_argument("--obj_pattern", type=str, default=None, help="Pattern with {scan_id} for object files.")
    parser.add_argument("--sg_pattern", type=str, default=None, help="Pattern with {scan_id} for scene-graph JSON files.")
    parser.add_argument("--scene_graph_pattern", type=str, default=None, help="Alias of --sg_pattern.")
    parser.add_argument("--edge_pattern", type=str, default=None, help="Alias of --sg_pattern for backward compatibility.")

    # matching / evaluation
    parser.add_argument(
        "--match_method",
        type=str,
        choices=["iou", "point_overlap"],
        default="iou",
        help="Object matching method. Default: iou.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.0,
        help="3D IoU threshold. Default 0.0. When threshold==0, the code uses positive-overlap semantics (IoU>0).",
    )
    parser.add_argument(
        "--eval_overlap_threshold",
        type=float,
        default=0.1,
        help="Radius used by point-overlap matching, kept from official ReplicaSSG evaluator.",
    )
    parser.add_argument("--skip_no_rel_objects", action="store_true")
    parser.add_argument("--use_aligned_ply", action="store_true")
    parser.add_argument(
        "--gt_mesh_pattern",
        type=str,
        default=None,
        help="Optional explicit GT mesh pattern, e.g. '/path/to/{scan_id}/labels.instances.annotated.v2.ply'",
    )

    # ranking
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1],
        help="Recall@k values for top-k retrieval metrics.",
    )

    # label embedding
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)

    # outputs
    parser.add_argument("--output_path", type=str, default=None, help="Optional text summary output path.")
    parser.add_argument("--summary_json", type=str, default=None, help="Optional JSON summary output path.")
    parser.add_argument(
        "--export_prediction_path",
        type=str,
        default=None,
        help="Optional pickle path to save converted predictions in official-style format.",
    )

    return parser.parse_args()


# ----------------------------- main ----------------------------- #

def main(args: argparse.Namespace) -> None:
    dataset_path = Path(args.root_path or args.dataset_path).expanduser().resolve()
    layout = resolve_dataset_layout(dataset_path)

    if args.dataset.lower() not in {"replicassg", "replica"}:
        print(f"[Info] --dataset={args.dataset} is accepted, but this script evaluates ReplicaSSG only.")

    scan_ids = determine_scan_ids(args, layout)
    result_specs = determine_result_specs(args, scan_ids)

    print(f"Target scans: {len(scan_ids)}")
    print(f"GT metadata dir: {layout.gt_files_dir}")
    print("Loading ReplicaSSG ground-truth metadata...")
    gt_bundle = load_replica_gt(layout, scan_ids, skip_no_rel_objects=args.skip_no_rel_objects)
    obj_classes = gt_bundle.class_mapping[OBJ_CLASS_NAME]
    rel_classes = gt_bundle.class_mapping[REL_CLASS_NAME]

    raw_results: Dict[str, Dict[str, Any]] = {}
    unique_obj_labels: set[str] = set()
    unique_rel_labels: set[str] = set()

    for scan_id in scan_ids:
        spec = result_specs.get(scan_id, {})
        obj_file = spec.get("obj_file")
        sg_file = spec.get("sg_file")

        if obj_file is None or sg_file is None:
            print(f"[Warning] Missing prediction paths for scan '{scan_id}'. Using empty prediction.")
            raw_results[scan_id] = {"objects": [], "scene_graph": {}}
            continue

        obj_file = Path(obj_file)
        sg_file = Path(sg_file)

        if not obj_file.exists() or not sg_file.exists():
            print(
                f"[Warning] Prediction file missing for scan '{scan_id}': "
                f"obj={obj_file.exists()} sg={sg_file.exists()}. Using empty prediction."
            )
            raw_results[scan_id] = {"objects": [], "scene_graph": {}}
            continue

        objects = load_prediction_objects(obj_file)
        scene_graph = load_scene_graph_json(sg_file)
        raw_results[scan_id] = {"objects": objects, "scene_graph": scene_graph}

        for obj_info in get_scene_graph_objects(scene_graph).values():
            if "label" in obj_info:
                unique_obj_labels.add(normalize_text(obj_info["label"]))
        for rel in get_scene_graph_relations(scene_graph):
            if "label" in rel:
                unique_rel_labels.add(normalize_text(rel["label"]))

    encoder_bank = LabelEncoderBank(
        obj_classes=obj_classes,
        rel_classes=rel_classes,
        device=args.device,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    encoder_bank.precompute(unique_obj_labels, unique_rel_labels)

    predictions: Dict[str, Dict[str, Any]] = {}
    print("Converting predictions to ReplicaSSG evaluation format...")
    for scan_id in tqdm(scan_ids, desc="Prediction conversion"):
        raw = raw_results.get(scan_id, {"objects": [], "scene_graph": {}})
        predictions[scan_id] = convert_scan_result_to_prediction_from_scene_graph(
            objects=raw["objects"],
            scene_graph=raw["scene_graph"],
            encoder_bank=encoder_bank,
            num_obj_classes=len(obj_classes),
            num_rel_classes=len(rel_classes),
        )

    if args.export_prediction_path:
        export_path = Path(args.export_prediction_path).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "wb") as f:
            pickle.dump(predictions, f)
        print(f"Saved converted predictions to: {export_path}")

    summary = evaluate_replica_predictions(
        layout=layout,
        gt_bundle=gt_bundle,
        predictions=predictions,
        match_method=args.match_method,
        iou_threshold=args.iou_threshold,
        eval_overlap_threshold=args.eval_overlap_threshold,
        ks=sorted(set(args.ks)),
        use_aligned_ply=args.use_aligned_ply,
        gt_mesh_pattern=args.gt_mesh_pattern,
    )

    # text = format_summary_text(summary)
    text = format_summary_text_simple(summary)
    print("\n" + text)

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text + "\n")
        print(f"Saved text summary to: {output_path}")

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON summary to: {summary_path}")


if __name__ == "__main__":
    main(parse_args())




"""
단일 scene:
python eval_replicassg.py \
  --dataset_path /home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG \
  --scene apartment_1 \
  --obj_file /home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG/apartment_1/CAPA_1/object/pcd_saves/full_pcd_ram_update.pkl.gz \
  --sg_file /home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG/apartment_1/CAPA_1/scene_graph/spatial_3d_scene_graph.json \
  --match_method iou \
  --iou_threshold 0.0 \
  --ks 1 \
  --summary_json /home/main/workspace/k2room2/CAPA-3DSG/replica_eval_apartment_1.json

test split 전체:
python eval_replicassg.py \
  --dataset_path /home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG \
  --split test \
  --obj_pattern "/home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG/{scan_id}/CAPA_1/object/pcd_saves/full_pcd_ram_update.pkl.gz" \
  --sg_pattern "/home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG/{scan_id}/CAPA_1/scene_graph/spatial_3d_scene_graph.json" \
  --match_method iou \
  --iou_threshold 0.0 \
  --ks 1 2 3 5 10 \
  --summary_json /home/main/workspace/k2room2/CAPA-3DSG/replica_eval_test.json

point-overlap로 바꾸려면:
python eval_replicassg.py \
  --dataset_path /home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG \
  --split test \
  --obj_pattern "/home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG/{scan_id}/CAPA_1/object/pcd_saves/full_pcd_ram_update.pkl.gz" \
  --sg_pattern "/home/main/workspace/k2room2/CAPA-3DSG/dataset/ReplicaSSG/{scan_id}/CAPA_1/scene_graph/spatial_3d_scene_graph.json" \
  --match_method point_overlap \
  --eval_overlap_threshold 0.1 \
  --ks 1
"""
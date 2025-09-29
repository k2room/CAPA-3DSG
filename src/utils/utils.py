import os
import copy
from pathlib import Path
import pickle
import gzip
import json

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

from ultralytics import YOLO, SAM as ULTRA_SAM
from groundingdino.util.inference import Model as GDINO
from segment_anything import sam_model_registry, SamPredictor
from ram.models import ram as RAM
from ram import inference_ram
import torchvision
import torchvision.transforms as TS


def _seg_sam(sam_pred: SamPredictor, bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    if len(xyxy) == 0:
        return np.empty((0, bgr.shape[0], bgr.shape[1]), dtype=bool)
    sam_pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = []
    for box in xyxy:
        masks, scores, _ = sam_pred.predict(box=box, multimask_output=True)
        out.append(masks[int(np.argmax(scores))])
    return np.asarray(out, dtype=bool)

def _ensure_bool_mask(m):
    return m.astype(bool) if m.dtype != bool else m

def _vis_safe_det(det: sv.Detections) -> sv.Detections:
    v = copy.copy(det)
    if getattr(v, "class_id", None) is None:
        return v
    cid = np.asarray(v.class_id).astype(int)
    if (cid < 0).any():
        base = int(cid[cid >= 0].max()) + 1 if (cid >= 0).any() else 0
        adj = cid.copy(); k = 0
        for i, c in enumerate(cid):
            if c < 0:
                adj[i] = base + k; k += 1
        v.class_id = adj
    return v

# --- RAM tag knowledge helpers ---

def load_knowledge(cfg):
    """Load object/part knowledge JSON. Try typo key then correct key."""
    p = getattr(cfg, "object_part_knowledge", None)
    if not p:
        return {"ram_add_obj": [], "ram_remove": [], "ram_remove_keyword": [], "small_object": [], "ram_add_part": {}}
    path = Path(p)
    if not path.exists():
        print(f"[WARN] object_part_knowledge not found: {path}")
        return {"ram_add_obj": [], "ram_remove": [], "ram_remove_keyword": [], "small_object": [], "ram_add_part": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize structure
    return {
        "ram_add_obj": list(data.get("ram_add_obj", [])),
        "ram_remove": list(data.get("ram_remove", [])),
        "ram_remove_keyword": list(data.get("ram_remove_keyword", [])),
        "small_object": list(data.get("small_object", [])),
        "ram_add_part": dict(data.get("ram_add_part", {})),
    }

def _curate_tags(raw_tags, knowledge, cfg) -> list[str]:
    """
    Apply remove/add/parts expansion to RAM tags.
    - Remove tags in ram_remove (exact match on normalized tag)
    - Remove tags containing any ram_remove_keyword (substring match)
    - Always add ram_add_obj and small_object
    - If a parent object from ram_add_part exists in current tags, add its parts
    - De-duplicate while preserving order
    """
    def norm(s): return str(s).strip().lower()
    
    # normalize knowledge
    rm_exact = {norm(x) for x in knowledge.get("ram_remove", []) if str(x).strip()}
    rm_sub   = [norm(x) for x in knowledge.get("ram_remove_keyword", []) if str(x).strip()]
    add_obj  = [str(x).strip() for x in (knowledge.get("ram_add_obj", []) + knowledge.get("small_object", [])) if str(x).strip()]
    add_part_map = {norm(k): [str(y).strip() for y in v] for k, v in knowledge.get("ram_add_part", {}).items()}

    # filter by remove
    # kept = [t for t in raw_tags if t and norm(t) not in rm]
    kept = []
    for t in raw_tags:
        if not t:
            continue
        tn = norm(t)
        # exact removal first
        if tn in rm_exact:
            continue
        # substring keyword removal
        if any(rm and rm in tn for rm in rm_sub):
            continue
        kept.append(t)

    # parts expansion conditioned on presence of parent in current tags
    cur_norm = {norm(t) for t in kept}
    parts_to_add = []
    for parent_norm, parts in add_part_map.items():
        if any(parent_norm in t for t in cur_norm):
            for p in parts:
                ps = parent_norm+" "+str(p).strip()
                if ps:
                    parts_to_add.append(ps)

    # always add add_obj
    out = []
    seen = set()
    for t in (kept + add_obj + parts_to_add):
        tt = str(t).strip()
        nl = norm(tt)
        if not tt:
            continue
        # optionally skip BG early if desired
        if getattr(cfg, "skip_bg", False) and nl in {str(x).lower() for x in getattr(cfg, "bg_classes", [])}:
            continue
        if nl not in seen:
            seen.add(nl)
            out.append(tt)
    return out


class DynamicClasses:
    def __init__(self, bg_classes=None, skip_bg: bool = False,
                 colors_file_path: str | Path | None = None, rng_seed: int | None = None):
        self._cls: list[str] = []
        self._lut: dict[str, int] = {}                 # lower -> idx
        self._bg = set([str(x).lower() for x in (bg_classes or [])])
        self.skip_bg = bool(skip_bg)

        self._colors: dict[str, list[float]] = {}      # class_name -> [r,g,b]
        self._colors_path: Path | None = None
        self._rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()

        if colors_file_path:
            p = Path(colors_file_path)
            if p.exists():
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            self._colors = {str(k): list(map(float, v)) for k, v in data.items()}
                    self._colors_path = p
                except Exception:
                    self._colors = {}
                    self._colors_path = p

    def _assign_color_if_needed(self, name: str):
        if name not in self._colors:
            self._colors[name] = self._rng.random(3).tolist()

    def _resolve_class_name(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self._cls):
                raise IndexError("Class index out of range.")
            return self._cls[key]
        elif isinstance(key, str):
            if key not in self._cls:
                raise ValueError(f"{key} is not a valid class name.")
            return key
        else:
            raise ValueError("Key must be an integer index or a string class name.")

    def add(self, words) -> bool:
        changed = False
        for w in words:
            t = str(w).strip()
            if not t:
                continue
            k = t.lower()
            if self.skip_bg and k in self._bg:
                continue
            if k not in self._lut:
                self._lut[k] = len(self._cls)
                self._cls.append(t)
                self._assign_color_if_needed(t)
                changed = True
        return changed

    def map(self, words) -> list[int]:
        out = []
        for w in words:
            k = str(w).strip().lower()
            out.append(self._lut.get(k, -1))
        return out

    def id_of(self, w) -> int:
        return self._lut.get(str(w).strip().lower(), -1)

    def get_classes_arr(self) -> list[str]:
        return list(self._cls)

    def get_bg_classes_arr(self) -> list[str]:
        return list(self._bg)

    def get_class_color(self, key) -> list[float]:
        name = self._resolve_class_name(key)
        return list(map(float, self._colors.get(name, [0.0, 0.0, 0.0])))

    def get_class_color_dict_by_index(self) -> dict[str, list[float]]:
        return {str(i): self.get_class_color(i) for i in range(len(self._cls))}

    def set_colors_path(self, path: str | Path):
        self._colors_path = Path(path)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for c in self._cls:
                f.write(f"{c}\n")
        if self._colors_path is None:
            self._colors_path = path.parent / f"{path.stem}_colors.json"
        with open(self._colors_path, "w", encoding="utf-8") as f:
            json.dump(self._colors, f)

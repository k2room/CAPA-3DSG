import os
import copy
from pathlib import Path
import pickle
import gzip
import json
from typing import List, Tuple
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


def load_knowledge(cfg):
    """
        Load object/part knowledge JSON. Try typo key then correct key.
    """
    p = getattr(cfg, "knowledge_path", None)
    if not p:
        print("[WARN] object_part_knowledge path not specified in config.")
        return {"ram_add_obj": [], "ram_remove": [], "ram_remove_keyword": [], "small_object": [], "ram_add_part": {}}

    path = Path(p)
    if not path.exists():
        print(f"[WARN] object_part_knowledge not found: {path}")
        return {"ram_add_obj": [], "ram_remove": [], "ram_remove_keyword": [], "small_object": [], "ram_add_part": {}}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "ram_add_obj": list(data.get("ram_add_obj", [])),
        "ram_remove": list(data.get("ram_remove", [])),
        "ram_remove_keyword": list(data.get("ram_remove_keyword", [])),
        "small_object": list(data.get("small_object", [])),
        "ram_add_part": dict(data.get("ram_add_part", {})),
    }

def curate_tags(raw_tags, knowledge, cfg) -> Tuple[List[str], List[str]]:
    """
        Apply remove/add/parts expansion to RAM tags, de-duplicating while preserving order
        - Remove tags in ram_remove (exact match on normalized tag)
        - Remove tags containing any ram_remove_keyword (substring match)
    """
    def norm(s): return str(s).strip().lower()
    
    bg_norm = set()
    if getattr(cfg, "skip_bg", False):
        bg_norm = {str(x).strip().lower() for x in getattr(cfg, "bg_classes", [])}

    # normalize knowledge
    rm_exact = {norm(x) for x in knowledge.get("ram_remove", []) if str(x).strip()}
    rm_sub   = [norm(x) for x in knowledge.get("ram_remove_keyword", []) if str(x).strip()]
    add_obj  = [str(x).strip() for x in knowledge.get("ram_add_obj", []) if str(x).strip()]
    small_obj  = [str(x).strip() for x in knowledge.get("small_object", []) if str(x).strip()]
    add_part_map = {norm(k): [str(y).strip() for y in v] for k, v in knowledge.get("ram_add_part", {}).items()}

    # filter by remove
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

    # always add add_obj
    out = []
    seen = set()
    for t in (kept + add_obj):
        tt = str(t).strip()
        nl = norm(tt)
        if not tt:
            continue
        # optionally skip BG early if desired
        if bg_norm and nl in bg_norm:
            continue
        if nl not in seen:
            seen.add(nl)
            out.append(tt)
    out.extend(small_obj)


    # parts expansion conditioned on presence of parent in current tags
    cur_norm = {norm(t) for t in kept}
    parts_to_add = []
    if cfg.use_part_knowledge:
        for parent_norm, parts in add_part_map.items():
            if any(parent_norm in t for t in cur_norm):
                for p in parts:
                    ps = parent_norm+":"+str(p).strip()
                    if ps:
                        parts_to_add.append(ps)
    parts_to_add.extend(small_obj)

    return out, parts_to_add
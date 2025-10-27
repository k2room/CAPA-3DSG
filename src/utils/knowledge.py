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

def curate_tags(raw_tags, knowledge, cfg) -> list[str]:
    """
        Apply remove/add/parts expansion to RAM tags, de-duplicating while preserving order
        - Remove tags in ram_remove (exact match on normalized tag)
        - Remove tags containing any ram_remove_keyword (substring match)
        - Always add ram_add_obj and small_object

        - If a parent object from ram_add_part exists in current tags, add its parts
        - 
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

    # always add add_obj
    out = []
    seen = set()
    for t in (kept + add_obj):
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


    # parts expansion conditioned on presence of parent in current tags
    cur_norm = {norm(t) for t in kept}
    parts_to_add = []
    for parent_norm, parts in add_part_map.items():
        if any(parent_norm in t for t in cur_norm):
            for p in parts:
                ps = parent_norm+" "+str(p).strip()
                if ps:
                    parts_to_add.append(ps)

    return out, parts_to_add
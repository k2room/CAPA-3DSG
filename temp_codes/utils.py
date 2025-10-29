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
import hydra
from omegaconf import DictConfig
import supervision as sv
from collections import Counter
import logging







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

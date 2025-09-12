#!/usr/bin/env python3
# import_test.py
import importlib, pkgutil, sys, os, subprocess

def probe(name, submod=None):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "no __version__")
        print(f"[OK] import {name}  (version: {ver})")
        if submod:
            importlib.import_module(f"{name}.{submod}")
            print(f"     └─ [OK] {name}.{submod}")
        return True
    except Exception as e:
        print(f"[FAIL] import {name}: {e.__class__.__name__}: {e}")
        return False

def main():
    # ------ Python / Torch / CUDA ------
    try:
        import torch
        print(f"python: {sys.version.split()[0]} | torch: {torch.__version__} (cuda {torch.version.cuda})")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    except Exception as e:
        print(f"[WARN] torch not importable: {e}")

    # nvcc 
    try:
        out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
        print("nvcc:", out.splitlines()[-1])
    except Exception:
        print("nvcc: not found or not callable")

    print("\n--- Probing third-party packages ---")

    # 1) Segment Anything
    probe("segment_anything")

    # 2) GroundingDINO + CUDA 
    if probe("groundingdino"):
        probe("groundingdino", "_C")  

    # 3) Recognize Anything (RAM/Tag2Text) 
    ram_found = False
    for cand in ["ram", "recognize_anything", "tag2text"]:
        if probe(cand):
            ram_found = True
            break
    if not ram_found:
        print("[WARN] RAM package not found (tried: ram, recognize_anything, tag2text)")

    # 4) Concept-Graphs
    cg_found = False
    for cand in ["conceptgraph", "conceptgraphs", "concept_graphs", "conceptgraph.conceptgraph"]:
        if probe(cand):
            cg_found = True
            break
    if not cg_found:
        print("[WARN] Concept-Graphs import not found. "
              "If using subtree only, ensure `pip install -e src/thirdparty/conceptgraph` "
              "or a sitecustomize/shim path is in place.")

def import_test():
    from ultralytics import YOLO, SAM as ULTRA_SAM
    from groundingdino.util.inference import Model as GDINO
    from segment_anything import sam_model_registry, SamPredictor
    from ram.models import ram as RAM
    from ram import inference_ram

    from conceptgraph.utils.general_utils import ObjectClasses
    from conceptgraph.slam.utils import filter_gobs

    print("\n[OK] All specific imports succeeded.")

if __name__ == "__main__":
    main()
    import_test()

""" 
$ python import_test.py 

python: 3.10.18 | torch: 2.0.1 (cuda 11.8)
torch.cuda.is_available(): True
nvcc: Build cuda_11.8.r11.8/compiler.31833905_0

--- Probing third-party packages ---
[OK] import segment_anything  (version: no __version__)
[OK] import groundingdino  (version: no __version__)
[OK] import groundingdino  (version: no __version__)
     └─ [OK] groundingdino._C
[OK] import ram  (version: no __version__)
[OK] import conceptgraph  (version: no __version__)

[OK] All specific imports succeeded.
"""

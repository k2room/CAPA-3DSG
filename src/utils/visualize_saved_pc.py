import argparse
import gzip
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from src.thirdparty.conceptgraph.conceptgraph.slam.slam_classes import (
    MapObjectList,
)

# Use VLPart's internal CLIP text encoder for part-level text embeddings
from src.utils.vlp_predictor import get_clip_embeddings


def _load_saved_results(pcd_path: Path):
    with gzip.open(pcd_path, "rb") as f:
        results = pickle.load(f)
    if not isinstance(results, dict):
        raise ValueError("Saved point cloud must be a dictionary pickle.")

    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    class_colors = results.get("class_colors", {})
    cfg = results.get("cfg", {})
    edges = results.get("edges", None)
    return objects, class_colors, cfg, edges


def _get_object_classes(objects: MapObjectList):
    # Most-common class id per object
    obj_classes = []
    for obj in objects:
        arr = np.asarray(obj["class_id"])  # list[int]
        if arr.size == 0:
            obj_classes.append(-1)
            continue
        vals, counts = np.unique(arr, return_counts=True)
        obj_classes.append(int(vals[np.argmax(counts)]))
    return obj_classes


def _apply_color_uniform(pcd: o3d.geometry.PointCloud, color_rgb):
    if isinstance(color_rgb, (list, tuple, np.ndarray)):
        color = np.asarray(color_rgb, dtype=float).reshape(1, 3)
    else:
        color = np.array([[float(color_rgb)] * 3])
    n = np.asarray(pcd.points).shape[0]
    if n == 0:
        return
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))


def visualize_saved_pc(pcd_path: Path):
    objects, class_colors, _cfg, _edges = _load_saved_results(pcd_path)

    # Prepare geometries
    pcds = [obj["pcd"] for obj in objects]
    bboxes = [obj["bbox"] for obj in objects]

    # Precompute class for color-by-class
    obj_classes = _get_object_classes(objects)

    # Setup visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Open3D - {pcd_path.name}", width=1280, height=720
    )
    for g in pcds + bboxes:
        vis.add_geometry(g)

    cmap = matplotlib.colormaps.get_cmap("turbo")

    # Flags
    visualize_saved_pc._show_bboxes = True
    # Original colors snapshot for 'R'
    orig_colors = [np.asarray(obj["pcd"].colors).copy() for obj in objects]

    def _update_all_pcds():
        for p in pcds:
            vis.update_geometry(p)

    # Callbacks
    def cb_color_by_class(_):
        for i, obj in enumerate(objects):
            cid = obj_classes[i]
            if cid < 0:
                continue
            col = class_colors.get(str(cid), [0.5, 0.5, 0.5])
            _apply_color_uniform(pcds[i], col)
        _update_all_pcds()

    def cb_color_by_rgb(_):
        for i, obj in enumerate(objects):
            p = pcds[i]
            p.colors = o3d.utility.Vector3dVector(orig_colors[i])
        _update_all_pcds()

    def cb_color_by_instance(_):
        if len(pcds) == 0:
            return
        cols = cmap(np.linspace(0, 1, len(pcds)))[:, :3]
        for i, p in enumerate(pcds):
            _apply_color_uniform(p, cols[i])
        _update_all_pcds()

    def cb_color_by_text_sim(_):
        # Query text → VLPart text encoder (part-level capable)
        try:
            text = input("Enter your part/object query: ").strip()
        except EOFError:
            print("[VLPart] No input received.")
            return
        if len(text) == 0:
            print("[VLPart] Empty query.")
            return

        with torch.no_grad():
            zs = get_clip_embeddings([text])  # (D, 1) CPU float32, L2-normalized
        text_ft = zs[:, 0]  # (D,)

        # Gather object features (already normalized in pipeline)
        obj_fts = objects.get_stacked_values_torch("clip_ft")  # (N, D)
        if obj_fts.ndim != 2 or obj_fts.shape[0] == 0:
            print("[VLPart] No object features available.")
            return

        # Cosine similarity
        sims = torch.nn.functional.cosine_similarity(
            text_ft.unsqueeze(0), obj_fts, dim=-1
        )  # (N,)
        sims_np = sims.cpu().numpy()
        if np.allclose(sims_np.max(), sims_np.min()):
            norm = np.zeros_like(sims_np)
        else:
            norm = (sims_np - sims_np.min()) / (sims_np.max() - sims_np.min())

        colors = cmap(norm)[:, :3]
        for i, p in enumerate(pcds):
            _apply_color_uniform(p, colors[i])
        _update_all_pcds()

        # Print top-1 info
        top_idx = int(np.argmax(sims_np))
        print(
            f"Top-1 match idx={top_idx}, class='{objects[top_idx]['class_name']}', sim={sims_np[top_idx]:.3f}"
        )

    def cb_toggle_bboxes(_):
        if visualize_saved_pc._show_bboxes:
            for b in bboxes:
                vis.remove_geometry(b, reset_bounding_box=False)
        else:
            for b in bboxes:
                vis.add_geometry(b, reset_bounding_box=False)
        visualize_saved_pc._show_bboxes = not visualize_saved_pc._show_bboxes

    # Register callbacks
    vis.register_key_callback(ord("C"), cb_color_by_class)
    vis.register_key_callback(ord("R"), cb_color_by_rgb)
    vis.register_key_callback(ord("I"), cb_color_by_instance)
    vis.register_key_callback(ord("F"), cb_color_by_text_sim)  # VLPart text encoder
    vis.register_key_callback(ord("B"), cb_toggle_bboxes)

    print("Key bindings: [C]lass, [R]GB, [I]nstance, [F]ind by text, [B]boxes")
    vis.run()


def main():
    ap = argparse.ArgumentParser(
        description="Visualize saved point cloud with part-level coloring and VLPart text queries"
    )
    ap.add_argument("--pcd_path", type=str, required=True, help="Path to pcd_*.pkl.gz")
    args = ap.parse_args()

    p = Path(args.pcd_path)
    if not p.exists():
        raise FileNotFoundError(f"pcd_path not found: {p}")
    visualize_saved_pc(p)


if __name__ == "__main__":
    main()

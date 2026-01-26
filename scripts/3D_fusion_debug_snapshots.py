"""3D Fusion using the detected objects/parts from each frame.

Debugging modifications (requested):
- Save intermediate fusion snapshots at 20/40/60/80/100% of the sequence as .pkl.gz.
- At each snapshot, save TWO files:
    (1) reconstructed: original per-point colors in each instance point cloud
    (2) instance-colored: each instance point cloud painted with its instance color

Output directory:
- Use `--save_path /path/to/output_dir`.
  If omitted, it falls back to the original default:
    <dataset_root>/<scene_id>/<save_folder_name>/(object|part)/pcd_saves/

Example:
    python scripts/3D_fusion_debug_snapshots.py \
        scene_id=0kitchen/video0 dataset=FunGraph3D save_folder_name=capa \
        --save_path /tmp/fusion_debug

You can also control snapshot downsample size (optional, Hydra override):
    snapshot_downsample_size=0.01
"""

# ===== imports =====
import copy
import gzip
import math
import pickle
import sys
from pathlib import Path
import warnings
import colorsys

import numpy as np
import open3d as o3d
import torch
import cv2
from tqdm import trange

# ===== hydra / omegaconf =====
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

# register custom resolver used in config strings: ${replace:x,/,-}
OmegaConf.register_new_resolver("replace", lambda s, a, b: str(s).replace(a, b))

warnings.simplefilter(action="ignore", category=FutureWarning)

# ===== project imports =====
from dataloader.datasets_common import get_dataset
from utils.ious import compute_2d_box_contained_batch
from utils.color_extraction import compute_color_sim
from slam.slam_classes import MapObjectList
from slam.utils import (
    create_or_load_colors,
    merge_obj2_into_obj1,
    denoise_objects,
    filter_objects,
    merge_objects,
    gobs_to_detection_list,
)
from slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
)

# Disable torch gradient computation
torch.set_grad_enabled(False)


# =========================
# Config enrichment
# =========================

def _resolve_path(p) -> str:
    """Resolve path against original CWD (Hydra changes run dir)."""
    pp = Path(str(p))
    if not pp.is_absolute():
        pp = Path(get_original_cwd()) / pp
    return str(pp)


def _process_cfg(cfg: DictConfig) -> None:
    """Attach dataset_root/dataset_config, and fill image sizes if missing."""
    if not cfg.get("scene_id") or not cfg.get("dataset"):
        raise ValueError(
            "Both `scene_id` and `dataset` are required. e.g., scene_id=0kitchen/video0 dataset=FunGraph3D"
        )
    if str(cfg.dataset) not in cfg.ALLOWED_DATASETS:
        raise ValueError(f"`dataset` must be one of {sorted(cfg.ALLOWED_DATASETS)}; got {cfg.dataset}")

    prev_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)

    ds = str(cfg.dataset)
    if ds == "FunGraph3D":
        cfg.dataset_root = _resolve_path(cfg.FUNGRAPH3D_root)
        cfg.dataset_config = _resolve_path(cfg.FUNGRAPH3D_config_path)
    elif ds == "SceneFun3Ddev":
        cfg.dataset_root = _resolve_path(Path(cfg.SCENEFUN3D_root) / "dev")
        cfg.dataset_config = _resolve_path(cfg.SCENEFUN3D_config)
    elif ds == "SceneFun3Dtest":
        cfg.dataset_root = _resolve_path(Path(cfg.SCENEFUN3D_root) / "test")
        cfg.dataset_config = _resolve_path(cfg.SCENEFUN3D_config)
    elif ds == "CAPAD":
        cfg.dataset_root = _resolve_path(cfg.CAPAD_root)
        cfg.dataset_config = _resolve_path(cfg.CAPAD_config)
    elif ds == "ReplicaSSG":
        cfg.dataset_root = _resolve_path(cfg.ReplicaSSG_root)
        cfg.dataset_config = _resolve_path(cfg.ReplicaSSG_config)
    else:
        raise ValueError(f"Unknown dataset: {ds}")

    dataset_cfg = OmegaConf.load(cfg.dataset_config)
    if ds == "CAPAD":
        cfg.image_height = 1440
        cfg.image_width = 1920
    else:
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width

    OmegaConf.set_struct(cfg, prev_struct)


# =========================
# Snapshot helpers
# =========================

def _normalize_color(color) -> list[float]:
    """Normalize various color formats to [r,g,b] floats in [0,1]."""
    if color is None:
        return [1.0, 1.0, 1.0]
    arr = np.asarray(color, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), constant_values=1.0)
    arr = arr[:3]
    if float(arr.max()) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr.tolist()


def _fallback_instance_color(i: int) -> list[float]:
    """Deterministic pseudo-random color (HSV golden-ratio)."""
    h = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return [float(r), float(g), float(b)]


def _strip_unneeded_keys(obj_list: MapObjectList) -> None:
    keep = {"pcd", "bbox", "clip_ft", "text_ft", "class_id", "num_detections", "inst_color"}
    for i in range(len(obj_list)):
        for k in list(obj_list[i].keys()):
            if k not in keep:
                del obj_list[i][k]


def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float = 0.01):
    """Prepare objects for saving (optionally voxel-downsample + keep only minimal keys)."""
    objects_to_save = copy.deepcopy(objects)

    if downsample_size is not None and float(downsample_size) > 0:
        ds = float(downsample_size)
        for i in range(len(objects_to_save)):
            if "pcd" in objects_to_save[i] and isinstance(objects_to_save[i]["pcd"], o3d.geometry.PointCloud):
                objects_to_save[i]["pcd"] = objects_to_save[i]["pcd"].voxel_down_sample(ds)

    _strip_unneeded_keys(objects_to_save)
    return objects_to_save.to_serializable()


def prepare_objects_save_instcolor(objects: MapObjectList, downsample_size: float = 0.01):
    """Prepare objects for saving, but paint each instance point cloud with an instance color."""
    objects_to_save = copy.deepcopy(objects)

    if downsample_size is not None and float(downsample_size) > 0:
        ds = float(downsample_size)
    else:
        ds = None

    for i in range(len(objects_to_save)):
        if "pcd" not in objects_to_save[i] or not isinstance(objects_to_save[i]["pcd"], o3d.geometry.PointCloud):
            continue

        if ds is not None:
            objects_to_save[i]["pcd"] = objects_to_save[i]["pcd"].voxel_down_sample(ds)

        inst_c = objects_to_save[i].get("inst_color", None)
        if inst_c is None:
            inst_c = _fallback_instance_color(i)
        color = _normalize_color(inst_c)
        objects_to_save[i]["pcd"].paint_uniform_color(color)

    _strip_unneeded_keys(objects_to_save)
    return objects_to_save.to_serializable()


def compute_color_similarities(cfg, detection_list, objects) -> torch.Tensor:
    """Compute color similarity matrix (M x N) from per-detection and per-object `color_feat` dicts."""
    M, N = len(detection_list), len(objects)
    if M == 0 or N == 0:
        return torch.zeros((M, N), dtype=torch.float32, device=cfg.device)

    det_feats = [detection_list[i].get("color_feat", None) for i in range(M)]
    obj_feats = [objects[j].get("color_feat", None) for j in range(N)]

    weights = getattr(cfg, "color_distance_weights", None)
    mapping = getattr(cfg, "color_sim_mapping", "inv")
    gamma = float(getattr(cfg, "color_sim_gamma", 3.0))

    S = compute_color_sim(det_feats, obj_feats, weights=weights, mapping=mapping, gamma=gamma)
    if not torch.is_tensor(S):
        S = torch.tensor(S, dtype=torch.float32)
    return S.to(device=cfg.device, dtype=torch.float32)


def aggregate_similarities_wc(
    cfg,
    spatial_sim: torch.Tensor,
    visual_sim: torch.Tensor,
    color_sim: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate spatial + visual (+ optional color) similarities."""
    device = visual_sim.device if torch.is_tensor(visual_sim) else spatial_sim.device
    spatial_sim = spatial_sim.to(device)
    visual_sim = visual_sim.to(device)
    if color_sim is not None:
        color_sim = color_sim.to(device)

    if str(getattr(cfg, "match_method", "sim_sum")) != "sim_sum":
        raise ValueError(f"Unknown matching method: {cfg.match_method}")

    pb = float(cfg.phys_bias)
    w_sp = 1.0 + pb
    w_vs = 1.0 - pb
    w_cl = float(cfg.w_color)
    sims = w_sp * spatial_sim + w_vs * visual_sim
    if color_sim is not None and w_cl != 0.0:
        sims = sims + w_cl * color_sim
    return sims


def _dump_gzip_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "wb") as f:
        pickle.dump(obj, f)


def _snapshot_stem(cfg, pct: int) -> str:
    stem = f"fusion_{pct:03d}pct_{cfg.gsa_variant}"
    if getattr(cfg, "use_color_feat", False):
        stem += "_wc"
    stem += "_part" if getattr(cfg, "part_reg", False) else "_obj"
    return stem


def _get_save_dir(cfg) -> Path:
    save_path = getattr(cfg, "save_path", None)
    if save_path is not None and str(save_path).strip() != "":
        out_dir = Path(_resolve_path(save_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # default (original)
    base = Path(cfg.dataset_root) / str(cfg.scene_id) / cfg.save_folder_name
    base = base / ("part" if cfg.part_reg else "object") / "pcd_saves"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _bg_dict_to_list(bg_objects):
    if bg_objects is None:
        return None
    if isinstance(bg_objects, dict):
        vals = [v for v in bg_objects.values() if v is not None]
        return MapObjectList(vals) if len(vals) > 0 else None
    return bg_objects


def _save_snapshot(
    cfg,
    pct: int,
    processed_frames: int,
    total_frames: int,
    objects: MapObjectList,
    bg_objects,
    classes,
    class_colors,
    save_dir: Path,
) -> None:
    ds = float(getattr(cfg, "snapshot_downsample_size", 0.01))

    bg_list = _bg_dict_to_list(bg_objects)

    exclude = set(getattr(cfg, "exclude_save_classes", ["wall","ceiling"]))  # e.g., ["wall","ceiling"]
    if exclude:
        objects_f = _filter_by_classname(objects, classes, exclude)
        bg_list_f = _filter_by_classname(bg_list, classes, exclude) if bg_list is not None else None
    else:
        objects_f, bg_list_f = objects, bg_list

    meta = {
        "scene_id": str(cfg.scene_id),
        "dataset": str(cfg.dataset),
        "part_reg": bool(getattr(cfg, "part_reg", False)),
        "progress": {
            "processed_frames": int(processed_frames),
            "total_frames": int(total_frames),
            "percent": int(pct),
        },
    }

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    stem = _snapshot_stem(cfg, pct)

    # (1) reconstructed snapshot
    recon = {
        **meta,
        "mode": "reconstructed",
        # "objects": prepare_objects_save_vis(objects, downsample_size=ds),
        # "bg_objects": None if bg_list is None else prepare_objects_save_vis(bg_list, downsample_size=ds),
        "objects": prepare_objects_save_vis(objects_f, downsample_size=ds),
        "bg_objects": None if bg_list_f is None else prepare_objects_save_vis(bg_list_f, downsample_size=ds),
        "cfg": cfg_dict,
        "class_names": classes,
        "class_colors": class_colors,
    }
    _dump_gzip_pickle(recon, save_dir / f"{stem}.pkl.gz")

    # (2) instance-colored snapshot
    inst = {
        **meta,
        "mode": "instance_colored",
        # "objects": prepare_objects_save_instcolor(objects, downsample_size=ds),
        # "bg_objects": None if bg_list is None else prepare_objects_save_instcolor(bg_list, downsample_size=ds),
        "objects": prepare_objects_save_instcolor(objects_f, downsample_size=ds),
        "bg_objects": None if bg_list_f is None else prepare_objects_save_instcolor(bg_list_f, downsample_size=ds),
        "cfg": cfg_dict,
        "class_names": classes,
        "class_colors": class_colors,
    }
    _dump_gzip_pickle(inst, save_dir / f"{stem}_instcolor.pkl.gz")


def _rewrite_save_path_flag_to_hydra_override() -> None:
    """Allow `--save_path ...` (or `--save_path=...`) while keeping Hydra overrides."""
    argv = sys.argv

    def _pop_kv(flag: str):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                raise ValueError(f"{flag} requires a value")
            v = argv[i + 1]
            del argv[i : i + 2]
            return v
        for i, a in enumerate(list(argv)):
            if a.startswith(flag + "="):
                v = a.split("=", 1)[1]
                del argv[i]
                return v
        return None

    save_path = _pop_kv("--save_path")
    if save_path is not None:
        # `++` lets us add the key even if it's not in the structured config.
        argv.append(f"++save_path={save_path}")


def _filter_by_classname(obj_list: MapObjectList, classes: list, exclude_names: set[str]) -> MapObjectList:
    """Drop items whose class_name (via class_id->classes) is in exclude_names."""
    if obj_list is None or len(obj_list) == 0:
        return obj_list
    kept = MapObjectList(device=obj_list)
    for i in range(len(obj_list)):
        cid = int(obj_list[i].get("class_id", [-1])[0])
        cname = classes[cid] if (0 <= cid < len(classes)) else None
        if cname is not None and cname in exclude_names:
            continue
        kept.append(obj_list[i])
    return kept


# =========================
# Main
# =========================

@hydra.main(version_base=None, config_path="../configs", config_name="CAPA_slam")
def main(cfg: DictConfig):
    _process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )

    # Load classes & colors
    classes, class_colors = create_or_load_colors(cfg, cfg.color_file_name)

    objects = MapObjectList(device=cfg.device)

    if not cfg.skip_bg:
        bg_objects = {c: None for c in cfg.bg_classes}
    else:
        bg_objects = None

    # Snapshot plan
    # snapshot_pcts = [20, 40, 60, 80, 100]
    snapshot_pcts = [1, 2, 3, 6, 12, 15, 50, 70, 90, 100]
    n_frames = len(dataset)
    frame_to_pct: dict[int, int] = {}
    for pct in snapshot_pcts:
        step = max(1, int(math.ceil(n_frames * pct / 100.0)))
        # if duplicates happen, keep the larger pct
        frame_to_pct[step] = max(frame_to_pct.get(step, 0), pct)

    saved_steps: set[int] = set()
    save_dir = _get_save_dir(cfg)

    for idx in trange(n_frames):
        # Load observations
        color_path = Path(dataset.color_paths[idx])
        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]
        image_rgb = color_tensor.cpu().numpy().astype(np.uint8)

        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()

        cam_K = intrinsics.cpu().numpy()[:3, :3]

        # Load detections
        if not cfg.part_reg:
            detections_path = (
                color_path.parent.parent
                / cfg.save_folder_name
                / "object"
                / cfg.detection_folder_name
                / color_path.name
            ).with_suffix(".pkl.gz")
        else:
            detections_path = (
                color_path.parent.parent
                / cfg.save_folder_name
                / "part"
                / cfg.detection_folder_name
                / color_path.name
            ).with_suffix(".pkl.gz")

        gobs = None
        try:
            with gzip.open(str(detections_path), "rb") as f:
                gobs = pickle.load(f)
        except Exception:
            gobs = None

        if gobs is not None:
            # Pose (untransformed)
            unt_pose = dataset.poses[idx].cpu().numpy()
            adjusted_pose = unt_pose

            fg_detection_list, bg_detection_list = gobs_to_detection_list(
                cfg=cfg,
                image=image_rgb,
                depth_array=depth_array,
                cam_K=cam_K,
                idx=idx,
                gobs=gobs,
                trans_pose=adjusted_pose,
                class_names=classes,
                BG_CLASSES=cfg.bg_classes,
                color_path=str(color_path),
                part_reg=cfg.part_reg,
            )

            # Background objects (fused per class)
            if bg_objects is not None and len(bg_detection_list) > 0:
                for detected_object in bg_detection_list:
                    class_name = detected_object["class_name"][0]
                    if bg_objects[class_name] is None:
                        bg_objects[class_name] = detected_object
                    else:
                        bg_objects[class_name] = merge_obj2_into_obj1(
                            cfg, bg_objects[class_name], detected_object, run_dbscan=False
                        )

            if len(fg_detection_list) > 0:
                # contain_number feature (optional)
                if cfg.use_contain_number:
                    xyxy = fg_detection_list.get_stacked_values_torch("xyxy", 0).to(cfg.device)
                    contain_numbers = compute_2d_box_contained_batch(xyxy, cfg.contain_area_thresh).to(cfg.device)
                    for i in range(len(fg_detection_list)):
                        fg_detection_list[i]["contain_number"] = [contain_numbers[i]]

                if len(objects) == 0:
                    for i in range(len(fg_detection_list)):
                        objects.append(fg_detection_list[i])
                else:
                    # similarity matrices
                    if cfg.part_reg and cfg.use_color_feat:
                        spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
                        visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
                        color_sim = compute_color_similarities(cfg, fg_detection_list, objects)
                        agg_sim = aggregate_similarities_wc(cfg, spatial_sim, visual_sim, color_sim)
                    else:
                        spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
                        visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
                        agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)

                    # contain_number mismatch penalty
                    if cfg.use_contain_number:
                        contain_numbers_objects = torch.tensor(
                            [obj["contain_number"][0] for obj in objects], device=cfg.device
                        )
                        detection_contained = (contain_numbers > 0).unsqueeze(1)  # (M,1)
                        object_contained = (contain_numbers_objects > 0).unsqueeze(0)  # (1,N)
                        xor = detection_contained ^ object_contained
                        agg_sim[xor] = agg_sim[xor] - cfg.contain_mismatch_penalty

                    # threshold
                    agg_sim[agg_sim < cfg.sim_threshold] = float("-inf")

                    # merge detections into objects
                    objects = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim)

                # periodic post-processing
                if cfg.denoise_interval > 0 and (idx + 1) % cfg.denoise_interval == 0:
                    objects = denoise_objects(cfg, objects)
                if cfg.filter_interval > 0 and (idx + 1) % cfg.filter_interval == 0:
                    objects = filter_objects(cfg, objects)
                if cfg.merge_interval > 0 and (idx + 1) % cfg.merge_interval == 0:
                    objects = merge_objects(cfg, objects)

        # ----- snapshots (always checked, even when no detections) -----
        step = idx + 1
        if step in frame_to_pct and step not in saved_steps:
            pct = frame_to_pct[step]
            _save_snapshot(
                cfg=cfg,
                pct=pct,
                processed_frames=step,
                total_frames=n_frames,
                objects=objects,
                bg_objects=bg_objects,
                classes=classes,
                class_colors=class_colors,
                save_dir=save_dir,
            )
            saved_steps.add(step)

    # Final post-processing (same as original)
    if bg_objects is not None:
        bg_objects = _bg_dict_to_list(bg_objects)
        if bg_objects is not None:
            bg_objects = denoise_objects(cfg, bg_objects)

    objects = denoise_objects(cfg, objects)

    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)

    # Optional: also save the final post-processed result (original behavior)
    if getattr(cfg, "save_pcd", False):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        name = f"full_pcd_{cfg.gsa_variant}"
        if getattr(cfg, "use_color_feat", False):
            name += "_wc"

        # Respect --save_path for the final output too
        final_dir = _get_save_dir(cfg)
        recon_post = {
            "scene_id": str(cfg.scene_id),
            "dataset": str(cfg.dataset),
            "mode": "reconstructed_post",
            "objects": objects.to_serializable(),
            "bg_objects": None if bg_objects is None else bg_objects.to_serializable(),
            "cfg": cfg_dict,
            "class_names": classes,
            "class_colors": class_colors,
        }
        _dump_gzip_pickle(recon_post, final_dir / f"{name}_post.pkl.gz")

        inst_post = {
            "scene_id": str(cfg.scene_id),
            "dataset": str(cfg.dataset),
            "mode": "instance_colored_post",
            "objects": prepare_objects_save_instcolor(objects, downsample_size=float(getattr(cfg, "snapshot_downsample_size", 0.01))),
            "bg_objects": None if bg_objects is None else prepare_objects_save_instcolor(bg_objects, downsample_size=float(getattr(cfg, "snapshot_downsample_size", 0.01))),
            "cfg": cfg_dict,
            "class_names": classes,
            "class_colors": class_colors,
        }
        _dump_gzip_pickle(inst_post, final_dir / f"{name}_post_instcolor.pkl.gz")


if __name__ == "__main__":
    _rewrite_save_path_flag_to_hydra_override()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    main()


"""
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE python scripts/3D_fusion_debug_snapshots.py \
  scene_id=4livingroom/video1 dataset=FunGraph3D save_folder_name=CAPA_1 \
  --save_path /home/main/workspace/k2room2/CAPA-3DSG/temp_codes/fusion_debug
"""
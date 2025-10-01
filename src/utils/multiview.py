from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Iterable, Tuple

import numpy as np
import open3d as o3d

from src.thirdparty.conceptgraph.conceptgraph.slam.slam_classes import MapObjectList


def _collect_scene_aabb(objects: Iterable[dict]) -> o3d.geometry.AxisAlignedBoundingBox:
    """Build a global AABB from object OBB corner points.

    Using OBB corners is fast and robust without touching large point buffers.
    """
    pts = []
    for obj in objects:
        try:
            corners = np.asarray(obj["bbox"].get_box_points())
            if corners.size:
                pts.append(corners)
        except Exception:
            pass
    if len(pts) == 0:
        # Fallback empty bbox centered at origin
        return o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    all_pts = np.vstack(pts)
    return o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(all_pts)
    )


def _add_objects_as_points(scene: o3d.visualization.rendering.Open3DScene, objects: Iterable[dict], point_size: float = 2.0):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"  # color from per-vertex color, no lighting needed
    mat.point_size = float(point_size)
    for i, obj in enumerate(objects):
        name = f"obj_pcd_{i}"
        try:
            scene.add_geometry(name, obj["pcd"], mat)
        except Exception:
            # Skip if geometry is invalid
            continue


def _setup_and_render(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    fov_deg: float,
    aspect: float,
    near: float,
    far: float,
    center: np.ndarray,
    eye: np.ndarray,
    up: np.ndarray,
):
    cam = renderer.scene.camera
    try:
        cam.set_projection(fov_deg, float(aspect), float(near), float(far))
    except TypeError:
        cam.set_projection(
            fov_deg,
            float(aspect),
            float(near),
            float(far),
            o3d.visualization.rendering.Camera.FovType.Vertical,
        )
    cam.look_at(center, eye, up)
    img = renderer.render_to_image()
    return img


def render_multiview_scene(
    objects: MapObjectList,
    out_dir: Path,
    *,
    obj_min_detections: int = 1,
    exclude_background: bool = True,
    image_size: Tuple[int, int] = (1600, 1200),  # (w, h)
    fov_deg: float = 100.0,
    radius_scale: float = 1.2,
    point_size: float = 2.0,
    include_bboxes: bool = False,
    color_mode: str = "instance",  # "rgb" | "class" | "instance"
    obj_classes = None,
):
    """Render front/back/left/right/top views of the fused 3D map.

    Args:
        objects: MapObjectList containing objects with 'pcd' and 'bbox'.
        out_dir: Directory to save images into (will create 'multiview' subdir).
        obj_min_detections: Filter threshold for stable objects.
        exclude_background: Whether to drop background objects for clarity.
        image_size: Output size (width, height).
        fov_deg: Camera field of view in degrees (wide angle recommended).
        radius_scale: Multiplier for camera distance from scene AABB extent.
        point_size: Point size for point cloud rendering.
        include_bboxes: If True, also overlay bounding boxes as line sets.
    """
    # Filter and color objects on a copy to avoid mutating pipeline state
    filtered = []
    for obj in objects:
        if obj.get("num_detections", 0) < obj_min_detections:
            continue
        if exclude_background and obj.get("is_background", False):
            continue
        o = deepcopy(obj)
        # Replace geometry handles with copies to avoid mutating the original map
        try:
            o["pcd"] = o3d.geometry.PointCloud(obj["pcd"])  # copy
        except Exception:
            pass
        try:
            bb_pts = np.asarray(obj["bbox"].get_box_points())
            o["bbox"] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(bb_pts)
            )
            try:
                o["bbox"].color = obj["bbox"].color
            except Exception:
                pass
        except Exception:
            pass
        filtered.append(o)
    objs_vis = MapObjectList(filtered)
    cm = str(color_mode or "").strip().lower()
    if cm == "rgb":
        # keep original per-point RGB colors as-is
        pass
    elif cm == "class" and obj_classes is not None:
        objs_vis.color_by_most_common_classes(obj_classes)
    else:
        objs_vis.color_by_instance()

    if len(objs_vis) == 0:
        # Nothing to render
        return

    aabb = _collect_scene_aabb(objs_vis)
    center = np.asarray(aabb.get_center())
    extent = np.asarray(aabb.get_extent())
    max_extent = float(np.max(extent)) if np.all(np.isfinite(extent)) else 1.0
    radius = max(1e-2, max_extent) * float(radius_scale)

    # Prepare offscreen renderer
    width, height = int(image_size[0]), int(image_size[1])
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white bg

    # Add geometries
    _add_objects_as_points(renderer.scene, objs_vis, point_size=point_size)
    if include_bboxes:
        mat_line = o3d.visualization.rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 1.0
        for i, obj in enumerate(objs_vis):
            try:
                ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obj["bbox"])  # type: ignore[arg-type]
                ls.paint_uniform_color(np.asarray(obj["bbox"].color))
                renderer.scene.add_geometry(f"bbox_{i}", ls, mat_line)
            except Exception:
                continue

    # Define views (Z is up). See docstring in commit message.
    up_horiz = np.array([0.0, 0.0, 1.0], dtype=float)
    up_top = np.array([0.0, 1.0, 0.0], dtype=float)

    extent_x, extent_y, extent_z = float(extent[0]), float(extent[1]), float(extent[2])
    height_z = max(1e-6, extent_z)
    top_offset = 0.375 * height_z

    views = {
        "front": {"eye": center.copy(), "target": center + np.array([0.0, +1.0, 0.0]), "up": up_horiz},
        "back":  {"eye": center.copy(), "target": center + np.array([0.0, -1.0, 0.0]), "up": up_horiz},
        "left":  {"eye": center.copy(), "target": center + np.array([-1.0, 0.0, 0.0]), "up": up_horiz},
        "right": {"eye": center.copy(), "target": center + np.array([+1.0, 0.0, 0.0]), "up": up_horiz},
        "top":   {"eye": center + np.array([0.0, 0.0, top_offset]), "target": center.copy(), "up": up_top},
    }

    out_dir = Path(out_dir) / "multiview"
    out_dir.mkdir(parents=True, exist_ok=True)

    aspect = (width / height) if height > 0 else 1.0
    near = max(1e-4, radius * 0.01)
    far = max(near * 10.0, radius * 50.0)

    for name, spec in views.items():
        eye = spec["eye"]
        target = spec["target"]
        up_vec = spec["up"]
        img = _setup_and_render(
            renderer,
            float(fov_deg),
            float(aspect),
            float(near),
            float(far),
            target,
            eye,
            up_vec,
        )
        o3d.io.write_image(str(out_dir / f"{name}.png"), img)

    # Some Open3D builds do not expose `release()` in Python; fall back to GC
    try:
        getattr(renderer, "release", None) and renderer.release()
    except Exception:
        pass
    del renderer

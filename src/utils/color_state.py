from __future__ import annotations
from typing import Any, Dict, List, Optional

from .color_extraction import ema_update_color_feat


class ColorFeatState:
    """
    Sidecar storage for per-object color features keyed by object id.
    Keeps thirdparty object/detection schemas untouched.
    """

    def __init__(self, ema_alpha: float = 0.30):
        self._map: Dict[Any, Optional[Dict[str, Any]]] = {}
        self.alpha = float(ema_alpha)

    def clear(self):
        self._map.clear()

    def get_for_object_id(self, obj_id: Any) -> Optional[Dict[str, Any]]:
        return self._map.get(obj_id, None)

    def get_obj_feat_list(self, objects: List[dict]) -> List[Optional[Dict[str, Any]]]:
        return [self._map.get(obj['id'], None) for obj in objects]

    def seed_from_detections(self, detection_list: List[dict], det_color_feats: List[Optional[Dict[str, Any]]]):
        for det, cf in zip(detection_list, det_color_feats):
            self._map[det['id']] = cf

    def update_post_merge(
        self,
        objects: List[dict],
        match_indices: List[Optional[int]],
        det_color_feats: List[Optional[Dict[str, Any]]],
        pre_len_objects: int,
    ):
        append_count = 0
        for d_idx, match in enumerate(match_indices):
            det_cf = det_color_feats[d_idx] if d_idx < len(det_color_feats) else None
            if det_cf is None:
                if match is None:
                    append_count += 1
                continue
            if match is None:
                obj_idx = pre_len_objects + append_count
                if obj_idx < len(objects):
                    obj_id = objects[obj_idx]['id']
                    self._map[obj_id] = det_cf
                append_count += 1
            else:
                obj_id = objects[match]['id']
                prev_cf = self._map.get(obj_id, None)
                self._map[obj_id] = ema_update_color_feat(prev_cf, det_cf, alpha=self.alpha) if prev_cf is not None else det_cf


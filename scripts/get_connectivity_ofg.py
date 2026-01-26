#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from collections import Counter

# Optional: if your project provides this, it will parse the serialized MapObjectList.
try:
    from slam.slam_classes import MapObjectList  # type: ignore
except Exception:
    MapObjectList = None  # type: ignore


AFFORDANCE_SYSTEM_PROMPT = """You are labeling an affordance for a given object part.

Describe the most likely human interaction in 1-3 words as a short verb phrase, focusing on the physical action.
Do NOT add any explanation. Do NOT output affordance label IDs (e.g., do not write "pinch_pull").

Affordance types (for reference only):
- rotate: adjusted by rotating a knob or dial
- key_press: consists of discrete keys or buttons to press
- tip_push: triggered by pushing with a fingertip
- hook_pull: pulled by hooking fingers
- pinch_pull: pulled via a pinch motion
- hook_turn: turned by hooking fingers and rotating
- foot_push: pushed using a foot
- plug_in: electrical power source or socket to insert a plug
- unplug: removing a plug from a socket
- etc: other plausible physical interactions not listed above are also allowed

If none of the affordance types clearly apply, output "none".

Return only the verb phrase.
"""


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_pickle(path: str) -> Any:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_list3(x: Any) -> Optional[List[float]]:
    if not (isinstance(x, (list, tuple)) and len(x) == 3):
        return None
    try:
        return [float(x[0]), float(x[1]), float(x[2])]
    except Exception:
        return None


def _safe_floats(x: Any) -> Optional[List[float]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


def _dist(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    if len(a) != 3 or len(b) != 3:
        return None
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def _round_floats(x: Any, ndigits: int = 3) -> Any:
    """Return a rounded copy for list/tuple of numbers; keep None/others as-is."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, (int, float)):
                out.append(round(float(v), ndigits))
            else:
                out.append(v)
        return out
    if isinstance(x, (int, float)):
        return round(float(x), ndigits)
    return x



def _ent_get(ent: Any, key: str, default: Any = None) -> Any:
    if isinstance(ent, dict):
        return ent.get(key, default)
    if hasattr(ent, "get"):
        try:
            return ent.get(key, default)
        except Exception:
            return default
    return default


def _bbox_center_extent(bbox: Any) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if bbox is None:
        return None, None

    if isinstance(bbox, dict):
        center = _safe_list3(bbox.get("center"))
        extent = _safe_floats(bbox.get("extent"))
        return center, extent

    center = None
    extent = None
    if hasattr(bbox, "center"):
        try:
            center = _safe_list3(list(getattr(bbox, "center")))
        except Exception:
            center = None
    if hasattr(bbox, "extent"):
        try:
            extent = _safe_floats(list(getattr(bbox, "extent")))
        except Exception:
            extent = None
    return center, extent


def _load_mol(serialized: Any) -> Any:
    # Keep robust: some pkls store list[dict] directly.
    if MapObjectList is None:
        return serialized
    try:
        mol = MapObjectList()
        mol.load_serializable(serialized)
        return mol
    except Exception:
        return serialized

def _center_extent_from_bbox_np(bbox_np: Any) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    bbox_np may be:
      - (8,3) corners
      - (2,3) min/max
      - flat length 6 [minx,miny,minz,maxx,maxy,maxz]
      - any Nx3 points
    We convert to AABB center/extent.
    """
    if bbox_np is None:
        return None, None

    # numpy -> list
    if hasattr(bbox_np, "tolist"):
        try:
            bbox_np = bbox_np.tolist()
        except Exception:
            pass

    minv = None
    maxv = None

    if isinstance(bbox_np, (list, tuple)) and len(bbox_np) > 0:
        # 2D list/tuple
        if isinstance(bbox_np[0], (list, tuple)):
            pts: List[List[float]] = []
            for p in bbox_np:
                if isinstance(p, (list, tuple)) and len(p) >= 3:
                    try:
                        pts.append([float(p[0]), float(p[1]), float(p[2])])
                    except Exception:
                        continue
            if len(pts) < 2:
                return None, None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            minv = [min(xs), min(ys), min(zs)]
            maxv = [max(xs), max(ys), max(zs)]

        # 1D flat
        else:
            flat: List[float] = []
            for v in bbox_np:
                try:
                    flat.append(float(v))
                except Exception:
                    continue

            if len(flat) == 6:
                minv = flat[:3]
                maxv = flat[3:6]
            elif len(flat) >= 6 and len(flat) % 3 == 0:
                pts = [flat[i : i + 3] for i in range(0, len(flat), 3)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                minv = [min(xs), min(ys), min(zs)]
                maxv = [max(xs), max(ys), max(zs)]
            else:
                return None, None
    else:
        return None, None

    center = [(minv[i] + maxv[i]) * 0.5 for i in range(3)]
    extent = [(maxv[i] - minv[i]) for i in range(3)]
    return center, extent


def _pick_label(ent: Any) -> str:
    """
    Prefer refined_obj_tag. If missing, fallback to class_name (mode if list).
    """
    tag = _ent_get(ent, "refined_obj_tag", None)
    if isinstance(tag, str) and tag.strip():
        return tag.strip()

    cn = _ent_get(ent, "class_name", None)
    if isinstance(cn, str) and cn.strip():
        return cn.strip()
    if isinstance(cn, (list, tuple)) and len(cn) > 0:
        vals = [str(x).strip() for x in cn if isinstance(x, str) and str(x).strip()]
        if vals:
            return Counter(vals).most_common(1)[0][0]

    # last fallback
    mc = _ent_get(ent, "majority_class_name", None)
    if isinstance(mc, str) and mc.strip():
        return mc.strip()

    return ""

def extract_nodes_from_pkl_results(pkl_results: Any, node_prefix: str) -> Dict[str, Dict[str, Any]]:
    """
    Reads bbox center/extent + label from full_pcd_ram_update.pkl(.gz).

    Real-world pkls often store:
      - pkl_results["objects"] as list[dict]
      - each dict has "bbox_np" and "class_name" (list), and refined_obj_tag may be missing.
    """
    if isinstance(pkl_results, dict) and "objects" in pkl_results:
        serialized = pkl_results["objects"]
    else:
        serialized = pkl_results

    mol = _load_mol(serialized)

    nodes: Dict[str, Dict[str, Any]] = {}

    # Case 1) list-like
    if isinstance(mol, list):
        for i, ent in enumerate(mol):
            bbox_np = _ent_get(ent, "bbox_np", None)
            if bbox_np is not None:
                center, extent = _center_extent_from_bbox_np(bbox_np)
            else:
                # fallback to old bbox format if present
                bbox = _ent_get(ent, "bbox", None)
                center, extent = _bbox_center_extent(bbox)

            label = _pick_label(ent)

            nodes[f"{node_prefix}_{i}"] = {
                "label": label,
                "center": center,
                "extent": extent,
            }
        return nodes

    # Case 2) MapObjectList-like (supports len + indexing)
    if hasattr(mol, "__len__") and hasattr(mol, "__getitem__"):
        n = len(mol)  # type: ignore
        for i in range(n):
            ent = mol[i]  # type: ignore
            bbox_np = _ent_get(ent, "bbox_np", None)
            if bbox_np is not None:
                center, extent = _center_extent_from_bbox_np(bbox_np)
            else:
                bbox = _ent_get(ent, "bbox", None)
                center, extent = _bbox_center_extent(bbox)

            label = _pick_label(ent)

            nodes[f"{node_prefix}_{i}"] = {
                "label": label,
                "center": center,
                "extent": extent,
            }
        return nodes

    raise ValueError("Unsupported node format in pkl results")


def extract_part_indices_from_edges(edges: Sequence[Tuple[int, int, int, str]]) -> List[int]:
    parts: Set[int] = set()
    for oi, pj, ok, lab in edges:
        # eval.py: part edge iff edge[2] == -1  -> part idx = edge[1]
        if int(ok) == -1 and int(pj) != -1:
            parts.add(int(pj))
    return sorted(parts)



def build_obj_candidates_for_part(
    part_idx: int,
    part_node: Dict[str, Any],
    edges: Sequence[Tuple[int, int, int, str]],
    obj_nodes: Dict[str, Dict[str, Any]],
    top_k: int,
) -> List[str]:
    cand_obj_idxs: Set[int] = set()
    for oi, pj, ok, lab in edges:
        oi = int(oi); pj = int(pj); ok = int(ok)
        # eval.py: part edge iff ok == -1, and then pj is the part index
        if ok == -1 and pj == part_idx:
            cand_obj_idxs.add(oi)

    cand_ids = [f"obj_{i}" for i in sorted(cand_obj_idxs) if f"obj_{i}" in obj_nodes]
    if not cand_ids:
        cand_ids = [oid for oid in obj_nodes.keys() if isinstance(oid, str) and oid.startswith("obj_")]

    # distance sort part_center -> obj_center logic stays the same
    p_center = part_node.get("center")
    if not (isinstance(p_center, list) and len(p_center) == 3):
        return cand_ids[:top_k] if top_k > 0 else cand_ids

    def key_fn(oid: str) -> float:
        o_center = obj_nodes.get(oid, {}).get("center", None)
        d = _dist(p_center, o_center)  # type: ignore
        return d if d is not None else float("inf")

    cand_ids = sorted(cand_ids, key=key_fn)
    return cand_ids[:top_k] if top_k > 0 else cand_ids


def _format_node_brief(node_id: str, node: Dict[str, Any], dist_val: Optional[float] = None) -> str:
    label = node.get("label", "")
    center = _round_floats(node.get("center", None), 3)
    extent = _round_floats(node.get("extent", None), 3)
    if dist_val is None:
        return f"- {node_id}: label={label}, center={center}, extent={extent}"
    return f"- {node_id}: label={label}, center={center}, extent={extent}, dist={dist_val:.4f}"


def ask_llm_part_of(
    client: Any,
    model: str,
    part_id: str,
    part_node: Dict[str, Any],
    candidates: List[Tuple[str, Dict[str, Any], Optional[float]]],
    store: bool = False,
) -> Optional[str]:
    """
    Returns an object id like "obj_12" or None.
    """
    sys_msg = (
        "You infer a PART-OF relation in a 3D indoor scene graph.\n"
        "Given one part node and a list of candidate object nodes, choose the SINGLE object that the part belongs to.\n"
        "Return ONLY a JSON object with the key 'object_id'.\n"
        "If none fits, set object_id to null.\n"
        "Do not include any other text."
    )

    part_desc = _format_node_brief(part_id, part_node)
    cand_lines = []
    for oid, onode, d in candidates:
        cand_lines.append(_format_node_brief(oid, onode, d))
    cand_text = "\n".join(cand_lines)

    user_msg = (
        f"Part node:\n{part_desc}\n\n"
        f"Candidate object nodes:\n{cand_text}\n\n"
        "Answer JSON only: {\"object_id\": \"obj_#\"} or {\"object_id\": null}"
    )

    print("=== LLM PART-OF QUERY ===")
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        # store=store,
    )
    print(resp)
    out = getattr(resp, "output_text", "") or ""
    out = out.strip()

    try:
        data = json.loads(out)
        if isinstance(data, dict):
            obj_id = data.get("object_id", None)
            if obj_id is None:
                return None
            if isinstance(obj_id, str) and re.fullmatch(r"obj_\d+", obj_id):
                return obj_id
    except Exception:
        pass

    m = re.search(r"\bobj_\d+\b", out)
    if m:
        return m.group(0)
    return None


def _clean_aff_phrase(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "none"
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    t = t.rstrip(".")
    return t if t else "none"


def ask_llm_affordance(
    client: Any,
    model: str,
    part_id: str,
    part_node: Dict[str, Any],
    parent_obj_id: Optional[str],
    parent_obj_node: Optional[Dict[str, Any]],
    store: bool = False,
) -> str:
    """
    Returns a short verb phrase (1–3 words) or "none".
    Stored as-is (no alias/ID normalization).
    """
    part_desc = _format_node_brief(part_id, part_node)

    if parent_obj_id and isinstance(parent_obj_node, dict):
        obj_desc = _format_node_brief(parent_obj_id, parent_obj_node)
        user_msg = f"Object part:\n{part_desc}\n\nLikely parent object:\n{obj_desc}\n"
    else:
        user_msg = f"Object part:\n{part_desc}\n"

    print("=== LLM AFFORDANCE QUERY ===")
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": AFFORDANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        # store=store, 
    )
    print(resp)
    out = getattr(resp, "output_text", "") or ""
    return _clean_aff_phrase(out)


def edges_to_functional_relations(edges: Sequence[Tuple[int, int, int, str]]) -> List[Dict[str, Any]]:
    rels: List[Dict[str, Any]] = []
    for oi, pj, ok, lab in edges:
        a = f"obj_{int(oi)}"
        b = f"part_{int(pj)}" if int(pj) != -1 else f"obj_{int(ok)}"
        rels.append({"pair": [a, b], "label": str(lab)})
    return rels


def gather_referenced_nodes(
    edges: Sequence[Tuple[int, int, int, str]],
    spatial_rel: Sequence[Dict[str, Any]],
) -> Tuple[Set[str], Set[str]]:
    objs: Set[str] = set()
    parts: Set[str] = set()

    for oi, pj, ok, lab in edges:
        oi = int(oi); pj = int(pj); ok = int(ok)

        if ok == -1:
            # part edge: (obj_i, part_j, -1, label)
            objs.add(f"obj_{oi}")
            if pj != -1:
                parts.add(f"part_{pj}")
        elif pj == -1:
            # obj-obj edge: (obj_i, -1, obj_k, label)
            objs.add(f"obj_{oi}")
            objs.add(f"obj_{ok}")
        else:
            # unexpected format: keep conservative
            objs.add(f"obj_{oi}")
            parts.add(f"part_{pj}")
            objs.add(f"obj_{ok}")

    # keep the spatial_rel merge as-is
    for r in spatial_rel:
        pair = r.get("pair", [])
        if isinstance(pair, list) and len(pair) == 2:
            a, b = pair[0], pair[1]
            if isinstance(a, str) and a.startswith("obj_"):
                objs.add(a)
            if isinstance(b, str) and b.startswith("part_"):
                parts.add(b)

    return objs, parts



def build_uni_graph(
    obj_nodes: Dict[str, Dict[str, Any]],
    part_nodes: Dict[str, Dict[str, Any]],
    edges: Sequence[Tuple[int, int, int, str]],
    spatial_rel: List[Dict[str, Any]],
    obj_to_parts: Dict[str, List[str]],
    part_afford: Dict[str, List[str]],
) -> Dict[str, Any]:
    referenced_objs, referenced_parts = gather_referenced_nodes(edges, spatial_rel)

    out_obj: Dict[str, Any] = {}
    out_part: Dict[str, Any] = {}

    def _idx(nid: str) -> int:
        try:
            return int(nid.split("_", 1)[1])
        except Exception:
            return 10**9

    for oid in sorted(referenced_objs, key=_idx):
        node = obj_nodes.get(oid, {})
        out_obj[oid] = {
            "label": node.get("label", ""),
            "center": node.get("center", None),
            "extent": node.get("extent", None),
            "connected_parts": sorted(list(set(obj_to_parts.get(oid, [])))),
        }

    for pid in sorted(referenced_parts, key=_idx):
        node = part_nodes.get(pid, {})
        out_part[pid] = {
            "label": node.get("label", ""),
            "center": node.get("center", None),
            "extent": node.get("extent", None),
            "affordance": part_afford.get(pid, []),
        }

    cfslam_edges_json = [[oi, pj, ok, lab] for (oi, pj, ok, lab) in edges]
    func_rel = edges_to_functional_relations(edges)

    return {
        "object": out_obj,
        "part": out_part,
        "cfslam_funcgraph_edges": cfslam_edges_json,
        "functional_relation": func_rel,
        "spatial_relation": spatial_rel,
    }

def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def ask_llm_part_of_batch(
    client: Any,
    model: str,
    items: List[Dict[str, Any]],
    store: bool = False,
) -> Dict[str, Optional[str]]:
    """
    One request for ALL part-of predictions.
    Output format:
      {"part_of": {"part_0": "obj_3", "part_1": null, ...}}
    """
    sys_msg = (
        "You infer PART-OF relations in a 3D indoor scene graph.\n"
        "For each item, choose the SINGLE best object candidate that the part belongs to.\n"
        "You MUST choose from the provided candidate object IDs.\n"
        "If none fits, return null.\n"
        "Return ONLY valid JSON in this format:\n"
        "{\"part_of\": {\"part_#\": \"obj_#\" or null, ...}}"
    )

    payload = {"items": items}
    user_msg = json.dumps(payload, ensure_ascii=False)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        # store=store, # turn off logs
    )

    out = getattr(resp, "output_text", "") or ""
    obj = _extract_json_obj(out)
    if not obj or "part_of" not in obj or not isinstance(obj["part_of"], dict):
        raise ValueError("Invalid LLM JSON for part_of batch")

    part_of: Dict[str, Optional[str]] = {}
    for pid, oid in obj["part_of"].items():
        if not isinstance(pid, str):
            continue
        if oid is None:
            part_of[pid] = None
            continue
        if isinstance(oid, str) and re.fullmatch(r"obj_\d+", oid):
            part_of[pid] = oid
        else:
            part_of[pid] = None
    return part_of


def ask_llm_affordance_batch(
    client: Any,
    model: str,
    items: List[Dict[str, Any]],
    store: bool = False,
) -> Dict[str, str]:
    """
    One request for ALL affordance predictions.
    Output format:
      {"affordance": {"part_0": "pull", "part_1": "rotate", ...}}
    (value is a short verb phrase OR "none")
    """
    sys_msg = (
        "You are labeling an affordance for a given object part.\n\n"
        "Describe the most likely human interaction in 1–3 words as a short verb phrase,\n"
        "focusing on the physical action.\n"
        "Do NOT add any explanation.\n"
        "Do NOT output affordance label IDs (e.g., do not write \"pinch_pull\").\n\n"
        "Affordance types (for reference only):\n"
        "- rotate: adjusted by rotating a knob or dial\n"
        "- key_press: consists of discrete keys or buttons to press\n"
        "- tip_push: triggered by pushing with a fingertip\n"
        "- hook_pull: pulled by hooking fingers\n"
        "- pinch_pull: pulled via a pinch motion\n"
        "- hook_turn: turned by hooking fingers and rotating\n"
        "- foot_push: pushed using a foot\n"
        "- plug_in: electrical power source or socket to insert a plug\n"
        "- unplug: removing a plug from a socket\n"
        "- etc: other plausible physical interactions not listed above are also allowed\n\n"
        "If none of the affordance types clearly apply, output \"none\".\n\n"
        "Return ONLY valid JSON in this format:\n"
        "{\"affordance\": {\"part_#\": \"verb phrase\" or \"none\", ...}}"
    )

    payload = {"items": items}
    user_msg = json.dumps(payload, ensure_ascii=False)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        # store=store, # turn off logs
    )

    out = getattr(resp, "output_text", "") or ""
    obj = _extract_json_obj(out)
    if not obj or "affordance" not in obj or not isinstance(obj["affordance"], dict):
        raise ValueError("Invalid LLM JSON for affordance batch")

    aff: Dict[str, str] = {}
    for pid, phrase in obj["affordance"].items():
        if not isinstance(pid, str):
            continue
        if not isinstance(phrase, str) or not phrase.strip():
            aff[pid] = "none"
            continue
        t = phrase.strip().split("\n", 1)[0].strip().rstrip(".")
        if not t:
            t = "none"
        # Hard truncate to first 3 words (safety)
        t = " ".join(t.split()[:3]) if t != "none" else "none"
        aff[pid] = t
    return aff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_pkl", required=True, help="Path to cfslam_funcgraph_edges.pkl (or .pkl.gz)")
    ap.add_argument("--obj_pkl", required=True, help="Path to object/full_pcd_ram_update.pkl(.gz)")
    ap.add_argument("--part_pkl", required=True, help="Path to part/full_pcd_ram_update.pkl(.gz)")

    ap.add_argument("--out_part_of", default="spatial_part_of.json", help="Output JSON for part-of relations")
    ap.add_argument("--out_uni_graph", default="uni_graph.json", help="Output unified graph JSON")

    ap.add_argument("--model", default="gpt-5", help="OpenAI model id")
    ap.add_argument("--top_k", type=int, default=20, help="Max candidate objects per part (distance-sorted)")
    ap.add_argument("--store", action="store_true", help="Store responses on OpenAI (default: false)")
    args = ap.parse_args()

    edges_raw = load_pickle(args.edges_pkl)
    if not isinstance(edges_raw, list):
        raise ValueError("edges_pkl must contain a list of (oi, pj, ok, label) tuples")

    edges: List[Tuple[int, int, int, str]] = []
    for e in edges_raw:
        if isinstance(e, tuple) and len(e) == 4:
            oi, pj, ok, lab = e
            edges.append((int(oi), int(pj), int(ok), str(lab)))

    print(f"Loaded {len(edges)} cfslam functional graph edges from {args.edges_pkl}")

    obj_results = load_pickle(args.obj_pkl)
    print(f"Loaded object results from {args.obj_pkl}")
    part_results = load_pickle(args.part_pkl)
    print(f"Loaded part results from {args.part_pkl}")

    # Node tables from PKL only (no init_graph_json)
    obj_nodes_all = extract_nodes_from_pkl_results(obj_results, "obj")
    part_nodes_all = extract_nodes_from_pkl_results(part_results, "part")

    # Restrict to nodes referenced in cfslam edges (as you requested)
    ref_objs, ref_parts = gather_referenced_nodes(edges, spatial_rel=[])
    obj_nodes: Dict[str, Dict[str, Any]] = {k: obj_nodes_all[k] for k in ref_objs if k in obj_nodes_all}
    part_nodes: Dict[str, Dict[str, Any]] = {k: part_nodes_all[k] for k in ref_parts if k in part_nodes_all}

    print(f"Extracted {len(obj_nodes)} object nodes and {len(part_nodes)} part nodes from PKL results")
    print(f"Referenced objects: {ref_objs}")
    print(f"Referenced parts: {ref_parts}")

    from openai import OpenAI  # noqa: E402
    OpenAI.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()

    part_indices = extract_part_indices_from_edges(edges)

    spatial_rel: List[Dict[str, Any]] = []
    obj_to_parts: Dict[str, List[str]] = {}
    part_afford: Dict[str, List[str]] = {}

    tmp_part_of = args.out_part_of + ".tmp"
    tmp_uni = args.out_uni_graph + ".tmp"

    def _save_progress():
        save_json(tmp_part_of, spatial_rel)
        uni = build_uni_graph(obj_nodes, part_nodes, edges, spatial_rel, obj_to_parts, part_afford)
        save_json(tmp_uni, uni)

    _save_progress()

    # try:
    #     for pi in part_indices:
    #         part_id = f"part_{pi}"
    #         part_node = part_nodes.get(part_id, None)
    #         if not isinstance(part_node, dict):
    #             continue

    #         # Part-of: choose among candidate objects from edges (and sorted by distance if possible)
    #         cand_obj_ids = build_obj_candidates_for_part(pi, part_node, edges, obj_nodes, top_k=args.top_k)
    #         candidates: List[Tuple[str, Dict[str, Any], Optional[float]]] = []

    #         p_center = part_node.get("center")
    #         for oid in cand_obj_ids:
    #             onode = obj_nodes.get(oid, {})
    #             d = _dist(p_center, onode.get("center")) if isinstance(p_center, list) else None
    #             candidates.append((oid, onode, d))

    #         chosen_obj: Optional[str] = None
    #         if candidates:
    #             try:
    #                 chosen_obj = ask_llm_part_of(
    #                     client=client,
    #                     model=args.model,
    #                     part_id=part_id,
    #                     part_node=part_node,  # includes extent from PKL
    #                     candidates=candidates,  # includes extent from PKL
    #                     store=args.store,
    #                 )
    #             except Exception as e:
    #                 print(f"[WARN] LLM(part-of) failed for {part_id}: {e}")

    #         cand_set = {oid for oid, _, _ in candidates}
    #         if chosen_obj is not None and chosen_obj in cand_set:
    #             spatial_rel.append({"pair": [chosen_obj, part_id], "label": "part of"})
    #             obj_to_parts.setdefault(chosen_obj, []).append(part_id)

    #         # Affordance: raw verb phrase (or "none") using your prompt
    #         try:
    #             parent_node = obj_nodes.get(chosen_obj, None) if chosen_obj else None
    #             aff_phrase = ask_llm_affordance(
    #                 client=client,
    #                 model=args.model,
    #                 part_id=part_id,
    #                 part_node=part_node,  # includes extent
    #                 parent_obj_id=chosen_obj if chosen_obj in cand_set else None,
    #                 parent_obj_node=parent_node if isinstance(parent_node, dict) else None,
    #                 store=args.store,
    #             )
    #         except Exception as e:
    #             print(f"[WARN] LLM(affordance) failed for {part_id}: {e}")
    #             aff_phrase = "none"

    #         part_afford[part_id] = [aff_phrase]

    #         _save_progress()

    try:
        # ---- Build batch input for PART-OF (one request) ----
        part_of_items: List[Dict[str, Any]] = []
        for pi in part_indices:
            part_id = f"part_{pi}"
            part_node = part_nodes.get(part_id, None)
            if not isinstance(part_node, dict):
                continue

            cand_obj_ids = build_obj_candidates_for_part(pi, part_node, edges, obj_nodes, top_k=args.top_k)

            candidates: List[Dict[str, Any]] = []
            p_center = part_node.get("center")
            for oid in cand_obj_ids:
                onode = obj_nodes.get(oid, {})
                if not isinstance(onode, dict):
                    continue
                d = _dist(p_center, onode.get("center")) if isinstance(p_center, list) else None
                candidates.append(
                    {
                        "id": oid,
                        "label": onode.get("label", ""),
                        "center": _round_floats(onode.get("center", None), 3),
                        "extent": _round_floats(onode.get("extent", None), 3),
                        "dist": round(d, 3) if isinstance(d, (int, float)) else d,  # dist도 줄이고 싶으면
                    }
                )

            part_of_items.append(
                {
                    "part_id": part_id,
                    "part": {
                        "label": part_node.get("label", ""),
                        "center": _round_floats(part_node.get("center", None), 3),
                        "extent": _round_floats(part_node.get("extent", None), 3),
                    },
                    "candidates": candidates,
                }
            )


        # ---- 1) Spatial relations: ONE LLM request ----
        part_of_map: Dict[str, Optional[str]] = {}
        if part_of_items:
            part_of_map = ask_llm_part_of_batch(
                client=client,
                model=args.model,
                items=part_of_items,
                store=args.store,
            )

        for part_id, obj_id in part_of_map.items():
            if obj_id is None:
                continue
            if part_id not in part_nodes:
                continue
            if obj_id not in obj_nodes:
                continue
            spatial_rel.append({"pair": [obj_id, part_id], "label": "part of"})
            obj_to_parts.setdefault(obj_id, []).append(part_id)

        _save_progress()

        # ---- 2) Affordances: ONE LLM request ----
        afford_items: List[Dict[str, Any]] = []
        for pi in part_indices:
            part_id = f"part_{pi}"
            part_node = part_nodes.get(part_id, None)
            if not isinstance(part_node, dict):
                continue

            obj_id = part_of_map.get(part_id, None)
            obj_node = obj_nodes.get(obj_id, None) if isinstance(obj_id, str) else None

            afford_items.append(
                {
                    "part_id": part_id,
                    "part": {
                        "label": part_node.get("label", ""),
                        "center": _round_floats(part_node.get("center", None), 3),
                        "extent": _round_floats(part_node.get("extent", None), 3),
                    },
                    "parent_object": (
                        {
                            "id": obj_id,
                            "label": obj_node.get("label", "") if isinstance(obj_node, dict) else "",
                            "center": _round_floats(obj_node.get("center", None), 3) if isinstance(obj_node, dict) else None,
                            "extent": _round_floats(obj_node.get("extent", None), 3) if isinstance(obj_node, dict) else None,
                        }
                        if isinstance(obj_id, str) and isinstance(obj_node, dict)
                        else None
                    ),
                }
            )


        aff_map: Dict[str, str] = {}
        if afford_items:
            aff_map = ask_llm_affordance_batch(
                client=client,
                model=args.model,
                items=afford_items,
                store=args.store,
            )

        for part_id, phrase in aff_map.items():
            part_afford[part_id] = [phrase]

        _save_progress()

    except KeyboardInterrupt:
        _save_progress()

    os.replace(tmp_part_of, args.out_part_of)
    os.replace(tmp_uni, args.out_uni_graph)

    print(f"Saved part-of relations: {args.out_part_of} (count={len(spatial_rel)})")
    print(f"Saved unified graph: {args.out_uni_graph}")
    print(f"Affordance inferred for parts: {len(part_afford)}")


if __name__ == "__main__":
    main()


"""
export OPENAI_API_KEY="YOUR_KEY"
python infer_part_of_and_build_uni.py \
  --edges_pkl /path/to/cfslam_funcgraph_edges.pkl \
  --init_graph_json /path/to/initial_3d_scene_graph.json \
  --out_part_of spatial_part_of.json \
  --out_uni_graph uni_graph.json \
  --model gpt-5 \
  --top_k 20

env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE python get_connectivity.py \
  --edges_pkl /home/main/workspace/k2room2/CAPA-3DSG/dataset/FunGraph3D/1bathroom/video0/gpt520250807/cfslam_funcgraph_edges.pkl \
  --obj_pkl /home/main/workspace/k2room2/gpuserver00_storage/CAPA/FunGraph3D/1bathroom/video0/gpt520250807/full_pcd_ram_withbg_allclasses_overlap_maskconf0_updated.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz \
  --part_pkl /home/main/workspace/k2room2/gpuserver00_storage/CAPA/FunGraph3D/1bathroom/video0/gpt520250807/full_pcd_ram_withbg_allclasses_overlap_maskconf0_updated.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz \
  --out_part_of /home/main/workspace/k2room2/CAPA-3DSG/dataset/FunGraph3D/1bathroom/video0/gpt520250807/spatial_part_of.json \
  --out_uni_graph /home/main/workspace/k2room2/CAPA-3DSG/dataset/FunGraph3D/1bathroom/video0/gpt520250807/uni_graph.json \
  --model gpt-5-2025-08-07 \
  --top_k 20  
"""
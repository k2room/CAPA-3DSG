#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import os
import pickle
import re
import shutil
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Optional: if your project provides this, it will parse the serialized MapObjectList.
try:
    from slam.slam_classes import MapObjectList  # type: ignore
except Exception:
    MapObjectList = None  # type: ignore


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
    if MapObjectList is None:
        return serialized
    try:
        mol = MapObjectList()
        mol.load_serializable(serialized)
        return mol
    except Exception:
        return serialized


def _center_extent_from_bbox_np(bbox_np: Any) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if bbox_np is None:
        return None, None

    if hasattr(bbox_np, "tolist"):
        try:
            bbox_np = bbox_np.tolist()
        except Exception:
            pass

    minv = None
    maxv = None

    if isinstance(bbox_np, (list, tuple)) and len(bbox_np) > 0:
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

    mc = _ent_get(ent, "majority_class_name", None)
    if isinstance(mc, str) and mc.strip():
        return mc.strip()

    return ""


def extract_nodes_from_pkl_results(pkl_results: Any, node_prefix: str) -> Dict[str, Dict[str, Any]]:
    if isinstance(pkl_results, dict) and "objects" in pkl_results:
        serialized = pkl_results["objects"]
    else:
        serialized = pkl_results

    mol = _load_mol(serialized)

    nodes: Dict[str, Dict[str, Any]] = {}

    if isinstance(mol, list):
        for i, ent in enumerate(mol):
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
        if ok == -1 and pj == part_idx:
            cand_obj_idxs.add(oi)

    cand_ids = [f"obj_{i}" for i in sorted(cand_obj_idxs) if f"obj_{i}" in obj_nodes]
    if not cand_ids:
        cand_ids = [oid for oid in obj_nodes.keys() if isinstance(oid, str) and oid.startswith("obj_")]

    p_center = part_node.get("center")
    if not (isinstance(p_center, list) and len(p_center) == 3):
        return cand_ids[:top_k] if top_k > 0 else cand_ids

    def key_fn(oid: str) -> float:
        o_center = obj_nodes.get(oid, {}).get("center", None)
        d = _dist(p_center, o_center)  # type: ignore
        return d if d is not None else float("inf")

    cand_ids = sorted(cand_ids, key=key_fn)
    return cand_ids[:top_k] if top_k > 0 else cand_ids


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
            objs.add(f"obj_{oi}")
            if pj != -1:
                parts.add(f"part_{pj}")
        elif pj == -1:
            objs.add(f"obj_{oi}")
            objs.add(f"obj_{ok}")
        else:
            objs.add(f"obj_{oi}")
            parts.add(f"part_{pj}")
            objs.add(f"obj_{ok}")

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
        store=store,
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
        store=store,
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
        t = " ".join(t.split()[:3]) if t != "none" else "none"
        aff[pid] = t
    return aff



# Heuristic pre-filter for object labels that are likely interactive parts (to reduce LLM load).
# We still let the LLM make the final decision.
_PART_HINT_RE = re.compile(
    r"(handle|knob|button|switch|lever|dial|latch|hinge|doorknob|door\s*knob|drawer\s*handle|cabinet\s*handle)",
    re.IGNORECASE,
)


def _looks_like_part_label(label: str) -> bool:
    t = (label or "").strip()
    if not t:
        return False
    return _PART_HINT_RE.search(t) is not None


def ask_llm_obj_as_part_batch(
    client: Any,
    model: str,
    items: List[Dict[str, Any]],
    store: bool = False,
) -> Dict[str, bool]:
    sys_msg = (
        "You are cleaning a 3D indoor scene graph before PART affordance and PART-OF inference.\n"
        "Some nodes in the 'object' list are actually object PARTS (components), e.g., handle/knob/button/switch/lever/dial.\n"
        "For each provided object, decide if it should ALSO be treated as a PART.\n"
        "Return true only when it is clearly a component of a larger object and typically not a standalone object.\n"
        "If ambiguous (e.g., door/drawer/cabinet/fridge), return false.\n\n"
        "Return ONLY valid JSON in this format:\n"
        '{"obj_as_part": {"obj_#": true/false, ...}}'
    )

    payload = {"items": items}
    user_msg = json.dumps(payload, ensure_ascii=False)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        store=store,
    )

    out = getattr(resp, "output_text", "") or ""
    obj = _extract_json_obj(out)
    if not obj or "obj_as_part" not in obj or not isinstance(obj["obj_as_part"], dict):
        raise ValueError("Invalid LLM JSON for obj_as_part batch")

    out_map: Dict[str, bool] = {}
    for oid, v in obj["obj_as_part"].items():
        if not (isinstance(oid, str) and re.fullmatch(r"obj_\d+", oid)):
            continue
        out_map[oid] = bool(v) if isinstance(v, bool) else False
    return out_map



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_pkl", required=True, help="Path to cfslam_funcgraph_edges_*.pkl (or .pkl.gz)")
    ap.add_argument("--nodes_pkl", required=True, help="Path to pkl.gz that contains object/part nodes")

    ap.add_argument("--out_part_of", default="spatial_part_of.json", help="Output JSON for part-of relations")
    ap.add_argument("--out_uni_graph", default="uni_graph.json", help="Output unified graph JSON")

    ap.add_argument("--model", default="gpt-5", help="OpenAI model id")
    ap.add_argument("--top_k", type=int, default=20, help="Max candidate objects per part (distance-sorted)")
    ap.add_argument("--store", action="store_true", help="Store responses on OpenAI (default: false)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_part_of) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_uni_graph) or ".", exist_ok=True)

    out_dir = os.path.dirname(args.out_uni_graph) or "."
    # try:
    #     shutil.copy2(args.edges_pkl, os.path.join(out_dir, os.path.basename(args.edges_pkl)))
    #     shutil.copy2(args.nodes_pkl, os.path.join(out_dir, os.path.basename(args.nodes_pkl)))
    # except Exception as e:
    #     print(f"[WARN] Failed to copy inputs: {e}")

    edges_raw = load_pickle(args.edges_pkl)
    if not isinstance(edges_raw, list):
        raise ValueError("edges_pkl must contain a list of (oi, pj, ok, label) tuples")

    edges: List[Tuple[int, int, int, str]] = []
    for e in edges_raw:
        if isinstance(e, tuple) and len(e) == 4:
            oi, pj, ok, lab = e
            edges.append((int(oi), int(pj), int(ok), str(lab)))

    print(f"Loaded {len(edges)} cfslam functional graph edges from {args.edges_pkl}")

    nodes_results = load_pickle(args.nodes_pkl)
    print(f"Loaded node results from {args.nodes_pkl}")

    obj_nodes_all: Dict[str, Dict[str, Any]] = {}
    part_nodes_all: Dict[str, Dict[str, Any]] = {}

    if isinstance(nodes_results, dict) and "objects" in nodes_results and "parts" in nodes_results:
        obj_nodes_all = extract_nodes_from_pkl_results({"objects": nodes_results["objects"]}, "obj")
        part_nodes_all = extract_nodes_from_pkl_results({"objects": nodes_results["parts"]}, "part")
    else:
        obj_nodes_all = extract_nodes_from_pkl_results(nodes_results, "obj")
        part_nodes_all = extract_nodes_from_pkl_results(nodes_results, "part")

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

    # --- Preprocess: some interactive parts may be misclassified under "object".
    # Keep obj_# as-is, but also copy into part_# (same index) before PART-OF + affordance inference.
    extra_part_indices: List[int] = []
    try:
        cand_items: List[Dict[str, Any]] = []
        for oid, onode in obj_nodes.items():
            if not isinstance(oid, str) or not re.fullmatch(r"obj_\d+", oid):
                continue
            if not isinstance(onode, dict):
                continue
            label = str(onode.get("label", "") or "")
            if _looks_like_part_label(label):
                cand_items.append(
                    {
                        "obj_id": oid,
                        "label": label,
                        "extent": _round_floats(onode.get("extent", None), 3),
                    }
                )

        if cand_items:
            obj_as_part = ask_llm_obj_as_part_batch(
                client=client,
                model=args.model,
                items=cand_items,
                store=args.store,
            )

            for oid, is_part in obj_as_part.items():
                if not is_part:
                    continue
                try:
                    idx = int(oid.split("_", 1)[1])
                except Exception:
                    continue

                pid = f"part_{idx}"
                if pid not in part_nodes:
                    src = obj_nodes.get(oid, {})
                    if isinstance(src, dict):
                        part_nodes[pid] = {
                            "label": src.get("label", ""),
                            "center": src.get("center", None),
                            "extent": src.get("extent", None),
                        }

                extra_part_indices.append(idx)

            if extra_part_indices:
                part_indices = sorted(set(part_indices).union(set(extra_part_indices)))
                uniq = sorted(set(extra_part_indices))
                print(
                    f"[Preprocess] Copied {len(uniq)} obj-nodes into part-nodes (kept obj_#): "
                    f"{uniq[:20]}" + (" ..." if len(uniq) > 20 else "")
                )
    except Exception as e:
        print(f"[WARN] Part-preprocess (obj->part copy) failed: {e}")


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

    try:
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
                        "dist": round(d, 3) if isinstance(d, (int, float)) else d,
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

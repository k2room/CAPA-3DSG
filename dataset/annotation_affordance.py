#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List, Any, Optional

AFFORDANCE_CHOICES = [
    "",  # blank allowed
    "rotate",
    "key_press",
    "tip_push",
    "hook_pull",
    "pinch_pull",
    "hook_turn",
    "foot_push",
    "plug_in",
    "unplug",
]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_tokens(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p != ""]


def prompt_affordances(annot_id: str, label: str, existing: Optional[List[str]] = None) -> List[str]:
    existing = existing or []
    existing_set = {x for x in existing if x in AFFORDANCE_CHOICES and x != ""}

    print(f"\n[Affordance Input] {label}  (annot_id={annot_id})")
    if existing_set:
        print(f"  - Existing: {sorted(existing_set)}")

    print("  - Allowed affordances:")
    for i, a in enumerate(AFFORDANCE_CHOICES):
        shown = "(blank)" if a == "" else a
        print(f"    {i}: {shown}")

    print("  - Input format: comma-separated labels OR comma-separated indices")
    print("    Examples: rotate,hook_turn   |   1,6   |   (empty to skip)")

    while True:
        s = input("  > ").strip()
        if s == "":
            return sorted(existing_set)

        toks = normalize_tokens(s)

        # Try indices first if all are digits
        if toks and all(t.isdigit() for t in toks):
            idxs = [int(t) for t in toks]
            if any(i < 0 or i >= len(AFFORDANCE_CHOICES) for i in idxs):
                print("  ! Invalid index. Try again.")
                continue
            selected = [AFFORDANCE_CHOICES[i] for i in idxs]
        else:
            selected = toks

        # Validate labels
        invalid = [x for x in selected if x not in AFFORDANCE_CHOICES]
        if invalid:
            print(f"  ! Invalid affordance(s): {invalid}. Try again.")
            continue

        selected_set = {x for x in selected if x != ""}
        merged = sorted(existing_set.union(selected_set))
        return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to FunGraph3D.annotations.v3.json")
    ap.add_argument("--rel", required=True, help="Path to FunGraph3D.relations.v2.json")
    ap.add_argument("--out", default=None, help="Output path for updated annotations JSON (default: <ann>.aff.json)")
    args = ap.parse_args()

    ann_list: List[Dict[str, Any]] = load_json(args.ann)
    rel_list: List[Dict[str, Any]] = load_json(args.rel)

    # Map annot_id -> annotation entry
    ann_map: Dict[str, Dict[str, Any]] = {}
    for a in ann_list:
        aid = a.get("annot_id")
        if isinstance(aid, str):
            ann_map[aid] = a

    # Cache user inputs to avoid prompting repeatedly
    afford_cache: Dict[str, List[str]] = {}

    out_path = args.out or (args.ann + ".aff.json")
    tmp_path = out_path + ".tmp"
    num = 0

    for r in rel_list:
        aid1 = r.get("first_node_annot_id")
        aid2 = r.get("second_node_annot_id")
        scene_id = r.get("scene_id", "")
        desc = r.get("description", "")

        if not isinstance(aid1, str) or not isinstance(aid2, str):
            continue

        a1 = ann_map.get(aid1)
        a2 = ann_map.get(aid2)
        if a1 is None or a2 is None:
            print(f"[WARN] Missing node in annotations for relation: {r.get('relation_id', '')}")
            continue

        l1 = a1.get("label", "<unknown>")
        l2 = a2.get("label", "<unknown>")

        # Print one-line pair info
        print(f"\n[{scene_id}] {l1} ({aid1})  <->  {l2} ({aid2})  |  {desc}")

        # Prompt affordances for node 1 (once)
        if aid1 not in afford_cache:
            existing1 = a1.get("affordance")
            existing1 = existing1 if isinstance(existing1, list) else []
            afford_cache[aid1] = prompt_affordances(aid1, str(l1), existing=existing1)

        # Prompt affordances for node 2 (once)
        if aid2 not in afford_cache:
            existing2 = a2.get("affordance")
            existing2 = existing2 if isinstance(existing2, list) else []
            afford_cache[aid2] = prompt_affordances(aid2, str(l2), existing=existing2)
        
        num += 1
        if num % 23 == 0:
            # Save temporary progress every 23 relations
            for aid, aff in afford_cache.items():
                if aid in ann_map:
                    ann_map[aid]["affordance"] = aff
            save_json(tmp_path, ann_list)

    # Write back to annotations
    for aid, aff in afford_cache.items():
        if aid in ann_map:
            ann_map[aid]["affordance"] = aff  # add/update field

    # out_path = args.out or (args.ann + ".aff.json")
    save_json(out_path, ann_list)
    print(f"\nDone. Saved updated annotations to: {out_path}")


if __name__ == "__main__":
    main()

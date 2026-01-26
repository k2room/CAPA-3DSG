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

def count_affordances(ann_list: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {a: 0 for a in AFFORDANCE_CHOICES if a != ""}

    for ann in ann_list:
        aff = ann.get("affordance", [])
        if not isinstance(aff, list):
            continue

        # Count unique labels per object (avoid double counting duplicates in one entry)
        for a in set(aff):
            if a in counts:
                counts[a] += 1

    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to FunGraph3D.annotations.v3.aff.json")
    args = ap.parse_args()

    ann_list: List[Dict[str, Any]] = load_json(args.ann)

    counts = count_affordances(ann_list)

    total_objects = len(ann_list)
    objects_with_any = sum(
        1
        for ann in ann_list
        if isinstance(ann.get("affordance", None), list) and any(x in counts for x in set(ann["affordance"]))
    )

    num = 0 
    print(f"Total objects: {total_objects}")
    print(f"Objects with >=1 affordance: {objects_with_any}")
    print("\nAffordance counts (number of objects having the label):")
    for k in AFFORDANCE_CHOICES:
        if k == "":
            continue
        print(f"  {k:10s}: {counts.get(k, 0)}")
        num += counts.get(k, 0)

    print("Total affordance counts:", num)


if __name__ == "__main__":
    main()

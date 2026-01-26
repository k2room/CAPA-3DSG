#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List


DELETE_LABELS = {
    "provide power",
    "remote to control",
    "turn on or turn off",
}

REPLACE_WITH = "part of"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_rel", required=True, help="Path to FunGraph3D.relations.v2.json")
    ap.add_argument("--out_rel", default="FunGraph3D.relations.v3.json", help="Output path")
    args = ap.parse_args()

    rel_list: List[Dict[str, Any]] = load_json(args.in_rel)

    new_list: List[Dict[str, Any]] = []
    removed = 0
    replaced = 0

    for r in rel_list:
        desc = r.get("description", "")
        if desc in DELETE_LABELS:
            removed += 1
            continue

        if "description" in r:
            r["description"] = REPLACE_WITH
            replaced += 1

        new_list.append(r)

    save_json(args.out_rel, new_list)

    print(f"Input relations: {len(rel_list)}")
    print(f"Removed (deleted labels): {removed}")
    print(f"Kept: {len(new_list)}")
    print(f"Replaced description -> '{REPLACE_WITH}': {replaced}")
    print(f"Saved: {args.out_rel}")


if __name__ == "__main__":
    main()

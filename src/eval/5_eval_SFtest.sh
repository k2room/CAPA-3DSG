#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT="/home/main/workspace/k2room2/CAPA-3DSG/src/eval/eval.py"

DATASET="SceneFun3D"
ROOT="/home/main/workspace/k2room2/gpuserver00_storage/CAPA/SceneFun3D_Graph"
SPLIT="test"
SAVE="CAPA_1"

# scene/video 목록
declare -a ITEMS=(
  "421380 42445022"
  "422391 42446522"
  "422813 42897545"  
  "460417 44358451"
  "460419 44358446"
  "422826 42897541"
  "466183 45260920"
  "466192 45260899"
  "466803 45261133"
  "467293 45261615"
  "468076 45261631"
  "469011 45663164"
)

for item in "${ITEMS[@]}"; do
  read -r SCENE VIDEO <<< "$item"

  OBJ_FILE="$ROOT/$SPLIT/$SCENE/$VIDEO/$SAVE/object/pcd_saves/full_pcd_ram_update.pkl.gz"
  PART_FILE="$ROOT/$SPLIT/$SCENE/$VIDEO/$SAVE/part/pcd_saves/full_pcd_ram_update.pkl.gz"
  EDGE_FILE="$ROOT/$SPLIT/$SCENE/$VIDEO/$SAVE/cfslam_funcgraph_edges.pkl"

  echo "Running: $SCENE / $VIDEO"
  "$PY" "$SCRIPT" \
    --dataset "$DATASET" \
    --root_path "$ROOT" \
    --scene "$SCENE" \
    --video "$VIDEO" \
    --split "$SPLIT" \
    --obj_file "$OBJ_FILE" \
    --part_file "$PART_FILE"\
    --edge_file "$EDGE_FILE"
done

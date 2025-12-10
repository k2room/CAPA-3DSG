#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT="/home/main/workspace/k2room2/CAPA-3DSG/src/eval/eval.py"

DATASET="SceneFun3D"
ROOT="/home/main/workspace/k2room2/gpuserver00_storage/CAPA/SceneFun3D_Graph"
SPLIT="dev"
SAVE="CAPA_1"

# scene/video 목록
declare -a ITEMS=(
  "420683 42445135"
  "421013 42444708"
  "421015 42444787"
  "421063 42444511"
  "421254 42444754"
  "421267 42444733"
  "421602 42445597"
  "422007 42446017"
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

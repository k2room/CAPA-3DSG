#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT="/home/main/workspace/k2room2/CAPA-3DSG/src/eval/eval.py"

DATASET="FunGraph3D"
ROOT="/home/main/workspace/k2room2/gpuserver00_storage/CAPA/FunGraph3D"
SAVE="CAPA_1"

# scene/video 목록
declare -a ITEMS=(
  "0kitchen video0"
  "1bathroom video0"
  "2livingroom video0"
  "3kitchen video0"
  "4livingroom video1"
  "5kitchen video0"
  "6kitchen video1"
  "7bedroom video0"
  "8bathroom video1"
  "9bedroom video1"
  "10kitchen video1"
  "11bedroom video1"
  "12kitchen video0"
  "13bathroom video0"
)

for item in "${ITEMS[@]}"; do
  read -r SCENE VIDEO <<< "$item"

  OBJ_FILE="$ROOT/$SCENE/$VIDEO/$SAVE/object/pcd_saves/full_pcd_ram_update.pkl.gz"
  PART_FILE="$ROOT/$SCENE/$VIDEO/$SAVE/part/pcd_saves/full_pcd_ram_update.pkl.gz"
  EDGE_FILE="$ROOT/$SCENE/$VIDEO/$SAVE/cfslam_funcgraph_edges.pkl"

  echo "Running: $SCENE / $VIDEO"
  "$PY" "$SCRIPT" \
    --dataset "$DATASET" \
    --root_path "$ROOT" \
    --scene "$SCENE" \
    --video "$VIDEO" \
    --obj_file "$OBJ_FILE" \
    --part_file "$PART_FILE"\
    --edge_file "$EDGE_FILE"
done

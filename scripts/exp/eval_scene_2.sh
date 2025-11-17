#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET="SceneFun3D"
SAVE="capa_1"
# "420683/42445135" "421013/42444708" "421015/42444787" "421063/42444511"
SCENES=("421254/42444754" "421267/42444733" "421602/42445597" "422007/42446017")

OBJ_FUSION_OPTS="mask_conf_threshold=0.30 max_bbox_area_ratio=0.90 merge_overlap_thresh=0.2 merge_visual_sim_thresh=0.6 merge_text_sim_thresh=0.8"
PART_FUSION_OPTS="mask_conf_threshold=0.15 max_bbox_area_ratio=0.15 merge_overlap_thresh=0.5 merge_visual_sim_thresh=0.75 merge_text_sim_thresh=0.7 part_reg=True"


run_cmd() {
  local scene="$1"; shift
  local cmd="$*"
  echo -e "\n==> SCENE=${scene}\nCMD: ${cmd}\n"
  ( cd "${ROOT}" && eval "${cmd}" ) 2>&1 
  local status=$?
  return $status
}

for scene in "${SCENES[@]}"; do
  run_cmd "${scene}" \
    python "${ROOT}/src/eval/eval_node.py" \
    --root_path /home/main/workspace/k2room2/gpuserver00_storage/CAPA/SceneFun3D_Graph --split dev \
    --dataset "${DATASET}" --scene "${scene}" --folder "${SAVE}"

done
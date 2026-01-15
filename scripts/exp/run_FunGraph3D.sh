#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET="FunGraph3D"
SAVE="CAPA_1"

SCENES=(
    "0kitchen/video0" 
    "1bathroom/video0" 
    "2livingroom/video0" 
    "3kitchen/video0" 
    "4livingroom/video1" 
    "5kitchen/video0" 
    "6kitchen/video1" 
    "7bedroom/video0" 
    "8bathroom/video1" 
    "9bedroom/video1" 
    "10kitchen/video1" 
    "11bedroom/video1" 
    "12kitchen/video0" 
    "13bathroom/video0"
    )

OBJ_FUSION_OPTS="mask_conf_threshold=0.30 max_bbox_area_ratio=0.90 merge_overlap_thresh=0.90 merge_visual_sim_thresh=0.75 merge_text_sim_thresh=0.70"
PART_FUSION_OPTS="mask_conf_threshold=0.15 max_bbox_area_ratio=0.15 merge_overlap_thresh=0.70 merge_visual_sim_thresh=0.70 merge_text_sim_thresh=0.70 part_reg=True"


RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT}/logs/${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

run_cmd() {
  local scene="$1"; shift
  local cmd="$*"
  local scene_sanitized="${scene//\//_}"
  local logfile="${LOG_DIR}/${scene_sanitized}.log"
  echo -e "\n==> SCENE=${scene}\nCMD: ${cmd}\n"
  ( cd "${ROOT}" && eval "${cmd}" ) 2>&1 | tee -a "${logfile}"
  return ${PIPESTATUS[0]}
}

for scene in "${SCENES[@]}"; do
  run_cmd "${scene}" \
    python "${ROOT}/scripts/2D_detection.py" \
      scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}"

  run_cmd "${scene}" \
    python "${ROOT}/scripts/3D_fusion.py" \
      scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}" \
      ${OBJ_FUSION_OPTS}

  run_cmd "${scene}" \
    python "${ROOT}/scripts/3D_fusion.py" \
      scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}" \
      ${PART_FUSION_OPTS}

  run_cmd "${scene}" \
    python "${ROOT}/scripts/gen_init_graph.py" \
      scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}"

  run_cmd "${scene}" \
    python "${ROOT}/scripts/gen_full_graph.py" \
      scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}"
done

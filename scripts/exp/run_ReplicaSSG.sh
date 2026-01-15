#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET="CAPAD"
SAVE="CAPA_1"

SCENES=( 
    "apartment_1/"
    "apartment_2/" 
    "frl_apartment_3/" 
    "frl_apartment_4/" 
    "frl_apartment_5/"
    "hotel_0/" 
    "office_1/"
    "office_3/"
    "office_4/" 
    "room_1/" 
    "room_2/"
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

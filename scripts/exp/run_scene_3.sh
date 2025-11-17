#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET="SceneFun3Dtest"
SAVE="capa_1"
#  "421380/42445022" "422391/42446522" "422813/42897545" "422826/42897541" "460417/44358451"
SCENES=("460419/44358446" "466183/45260920" "466192/45260899" "466803/45261133" "467293/45261615" "468076/45261631" "469011/45663164")

OBJ_FUSION_OPTS="mask_conf_threshold=0.30 max_bbox_area_ratio=0.90 merge_overlap_thresh=0.2 merge_visual_sim_thresh=0.6 merge_text_sim_thresh=0.8"
PART_FUSION_OPTS="mask_conf_threshold=0.15 max_bbox_area_ratio=0.15 merge_overlap_thresh=0.5 merge_visual_sim_thresh=0.75 merge_text_sim_thresh=0.7 part_reg=True"

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

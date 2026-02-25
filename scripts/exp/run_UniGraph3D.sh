#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# =====================
# Config (override via env if needed)
# =====================
SAVE="${SAVE:-CAPA_woc_2}"

OBJ_FUSION_OPTS="${OBJ_FUSION_OPTS:-mask_conf_threshold=0.30 max_bbox_area_ratio=0.90 merge_overlap_thresh=0.90 merge_visual_sim_thresh=0.75 merge_text_sim_thresh=0.70}"
PART_FUSION_OPTS="${PART_FUSION_OPTS:-mask_conf_threshold=0.15 max_bbox_area_ratio=0.15 merge_overlap_thresh=0.70 merge_visual_sim_thresh=0.70 merge_text_sim_thresh=0.70 part_reg=True}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# If set to 1, stop the whole run on first failure
FAIL_FAST="${FAIL_FAST:-0}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT}/logs/${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

SUMMARY_LOG="${LOG_DIR}/summary.log"
touch "${SUMMARY_LOG}"

run_cmd() {
  local dataset="$1"; shift
  local scene="$1"; shift
  local cmd="$*"

  local scene_sanitized="${scene//\//_}"
  local ds_dir="${LOG_DIR}/${dataset}"
  mkdir -p "${ds_dir}"

  local logfile="${ds_dir}/${scene_sanitized}.log"

  echo -e "\n==> DATASET=${dataset}  SCENE=${scene}\nCMD: ${cmd}\nlog file: ${logfile}\n"
  ( cd "${ROOT}" && eval "${cmd}" ) 2>&1 | tee -a "${logfile}"
  local rc=${PIPESTATUS[0]}

  if [[ ${rc} -ne 0 ]]; then
    echo "[FAIL] dataset=${dataset} scene=${scene} rc=${rc}" | tee -a "${SUMMARY_LOG}"
    if [[ "${FAIL_FAST}" == "1" ]]; then
      echo "FAIL_FAST=1 -> exiting." | tee -a "${SUMMARY_LOG}"
      exit ${rc}
    fi
  else
    echo "[ OK ] dataset=${dataset} scene=${scene}" >> "${SUMMARY_LOG}"
  fi

  return ${rc}
}

# =====================
# Scene lists
# =====================
SCENES_FunGraph3D=(
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

SCENES_SceneFun3Ddev=(
  "420683/42445135"
  "421013/42444708"
  "421015/42444787"
  "421063/42444511"
  "421254/42444754"
  "421267/42444733"
  "421602/42445597"
  "422007/42446017"
)

SCENES_SceneFun3Dtest=(
  "421380/42445022"
  "422391/42446522"
  "422813/42897545"
  "422826/42897541"
  "460417/44358451"
  "460419/44358446"
  "466183/45260920"
  "466192/45260899"
  "466803/45261133"
  "467293/45261615"
  "468076/45261631"
  "469011/45663164"
)

run_dataset() {
  local dataset="$1"; shift

  local -n scenes_ref="$1"

  echo -e "\n=============================="
  echo " Running DATASET=${dataset}"
  echo " SAVE=${SAVE}"
  echo " LOG_DIR=${LOG_DIR}/${dataset}"
  echo "=============================="

  for scene in "${scenes_ref[@]}"; do
    run_cmd "${dataset}" "${scene}" \
      env OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
      python "${ROOT}/scripts/2D_detection.py" \
        scene_id="${scene}" dataset="${dataset}" save_folder_name="${SAVE}"

    run_cmd "${dataset}" "${scene}" \
      env OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
      python "${ROOT}/scripts/3D_fusion.py" \
        scene_id="${scene}" dataset="${dataset}" save_folder_name="${SAVE}" \
        ${OBJ_FUSION_OPTS}

    run_cmd "${dataset}" "${scene}" \
      env OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
      python "${ROOT}/scripts/3D_fusion.py" \
        scene_id="${scene}" dataset="${dataset}" save_folder_name="${SAVE}" \
        ${PART_FUSION_OPTS}

    run_cmd "${dataset}" "${scene}" \
      env OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
      python "${ROOT}/scripts/gen_init_graph.py" \
        scene_id="${scene}" dataset="${dataset}" save_folder_name="${SAVE}"

    run_cmd "${dataset}" "${scene}" \
      env OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
      python "${ROOT}/scripts/gen_full_graph.py" \
        scene_id="${scene}" dataset="${dataset}" save_folder_name="${SAVE}"
  done
}

# =====================
# Run all datasets
# =====================
run_dataset "FunGraph3D" SCENES_FunGraph3D
run_dataset "SceneFun3Ddev" SCENES_SceneFun3Ddev
run_dataset "SceneFun3Dtest" SCENES_SceneFun3Dtest

echo -e "\nAll done. Summary: ${SUMMARY_LOG}\n"
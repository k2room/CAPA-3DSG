#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET="CAPAD"
SAVE="CAPA_1"


# scene0/42445173
# scene1/43649409
# scene2/47331576
# scene3/47334499
# scene4/42444755
# scene5/42445078
# scene6/42445692
# scene7/42444514
# scene8/42444887
# scene9/42444490
# scene10/42447294
# scene11/42445931
# scene12/42897405
# scene13/42447329
# scene14/47334559
# scene15/42898221
# scene16/video0
# scene17/video1
# scene18/video0
# scene19/video1
# scene20/video1
# scene21/video0
# scene22/video1
# scene23/video0
# scene24/video0
# scene25/video0
# scene26/video0
# scene27/video0


SCENES=( 
    # "scene0/42445173"
    # "scene1/43649409"
    # "scene2/47331576"
    # "scene3/47334499"
    # "scene4/42444755"
    # "scene5/42445078"
    # "scene6/42445692"
    # "scene7/42444514"
    # "scene8/42444887"
    # "scene9/42444490"
    # "scene10/42447294"
    # "scene11/42445931"
    # "scene12/42897405"
    # "scene13/42447329"
    # "scene14/47334559"
    # "scene15/42898221"
    "scene16/video0"
    # "scene17/video1"
    # "scene18/video0"
    # "scene19/video1"
    # "scene20/video1"
    # "scene21/video0"
    # "scene22/video1"        
    # "scene23/video0"
    # "scene24/video0"
    # "scene25/video0"
    # "scene26/video0"
    # "scene27/video0"
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
#   run_cmd "${scene}" \
#     python "${ROOT}/scripts/2D_detection.py" \
#       scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}"

#   run_cmd "${scene}" \
#     python "${ROOT}/scripts/3D_fusion.py" \
#       scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}" \
#       ${OBJ_FUSION_OPTS}

#   run_cmd "${scene}" \
#     python "${ROOT}/scripts/3D_fusion.py" \
#       scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}" \
#       ${PART_FUSION_OPTS}

#   run_cmd "${scene}" \
#     python "${ROOT}/scripts/gen_init_graph.py" \
#       scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}"

  run_cmd "${scene}" \
    python "${ROOT}/scripts/gen_full_graph.py" \
      scene_id="${scene}" dataset="${DATASET}" save_folder_name="${SAVE}"
done

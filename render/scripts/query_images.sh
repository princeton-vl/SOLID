#!/bin/bash
#
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=1

source ./scripts/setup_paths.sh

object_json="${1}"
dataset_path="${2}"
start_idx="${3:-0}"

workerid="${SLURM_ARRAY_TASK_ID:-1}"
task_index="$((start_idx + workerid - 1))"

blenderproc \
    run \
    --custom-blender-path "${BLENDER_PATH}" \
    --temp-dir "${TEMP_DIR}" \
    ./rendering/main.py \
    --index "${task_index}" \
    --model_dir "${MODEL_DIR}" \
    --scenenet_dir "${SCENENET_DIR}" \
    --scenenet_texture_dir "${SCENENET_TEXTURE_DIR}" \
    --target queries \
    query \
    --angle_delta 45 \
    --min_thetax -30 \
    --max_thetax -30 \
    "${object_json}" \
    "${dataset_path}" 

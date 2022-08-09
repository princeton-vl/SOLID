#!/bin/bash

INP_DIR="${1:-ShapeNetCore.v2}"
OUP_DIR="${2:-shapenet}"

BLENDER_DIR='../blender' 

NUM_PROCS=$(nproc)

function convert() {
    hash_id="${1}"
    inp_dir="${2}"
    oup_dir="${3}"

    blender_dir="${4}"

    if [ -f "${oup_dir}/${hash_id}/model.glb" ]; then
        return
    fi

    mkdir -p "${oup_dir}/${hash_id}"
    obj2gltf -i "${inp_dir}/${hash_id}/models/model_normalized.obj" -o "${oup_dir}/${hash_id}/model.gltf"
    blenderproc run \
        --custom-blender-path "${blender_dir}" \
        bpy_clean_mesh.py \
        --source_path "${oup_dir}/${hash_id}/model.gltf" \
        --target_path "${oup_dir}/${hash_id}/model.glb" \
        --asset_id "${hash_id}"
}

export -f convert

if [ ! -d "${INP_DIR}" ]; then
    echo "Cannot find ShapeNet. ${INP_DIR} does not exist."
    exit -1
fi

if [ ! -d "${BLENDER_DIR}" ]; then
    echo "Cannot find Blender. ${BLENDER_DIR} does not exist."
    exit -1
fi

mkdir -p "${OUP_DIR}"
cat shapenet_ids.txt | xargs -n 1 -I {} -P ${NUM_PROCS} bash -c 'convert "$@"' _ {} "${INP_DIR}" "${OUP_DIR}" "${BLENDER_DIR}"

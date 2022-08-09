#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:${PWD}"
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NCCL_IB_DISABLE=2

usage() {
    echo "Usage: ${0} [-p PORT] [-h HOST] [-g NUM_GPUS] CONFIG" 1>&2;
    exit -1;
}

declare port=9896
declare host='127.0.0.1'
declare num_gpus=$(nvidia-smi --list-gpus | wc -l)

while getopts ":p:g:h:" op; do
    case "${op}" in
        p) port=${OPTARG};;
        h) host=${OPTARG};;
        g) num_gpus=${OPTARG};;
        \?) echo "Invalid option: -${OPTARG}" >&2
            exit -1;;
    esac
done

shift $((OPTIND - 1))
if [ -z "${1}" ]; then
    usage
fi

config="${1}"

if [ ! -z ${SLURM_JOB_ID} ]; then
    echo "[$(date)] Script is running in SLURM"
    _host=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
    echo "[$(date)] Replace ${host} with ${_host}"
    host=${_host}
fi

./tools/lazyconfig_train_net.py \
    --config-file "${config}" \
    --num-gpus ${num_gpus} \
    --num-machines "${SLURM_JOB_NUM_NODES:-1}" \
    --machine-rank "${SLURM_PROCID:-0}" \
    --dist-url "tcp://${host}:${port}" \
    --resume \
        "${@:2}"

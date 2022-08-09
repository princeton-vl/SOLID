#!/bin/bash
#
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1

export PYTHONPATH="${PYTHONPATH}:${PWD}"

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

num_gpus=$(nvidia-smi --list-gpus | wc -l)
config="${1}"

if [ ! -z ${SLURM_JOB_ID} ]; then
    echo "[$(date)] Script is running in SLURM"
    _host=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
    echo "[$(date)] Replace ${host} with ${_host}"
    host=${_host}
fi

./tools/train_net.py \
    --config-file "${config}" \
    --num-gpus ${num_gpus} \
    --dist-url "tcp://${host}:${port}" \
    --resume \
        MODEL.BACKBONE.FREEZE_AT 0 \
        SOLVER.AMP.ENABLED True \
        "${@:2}"

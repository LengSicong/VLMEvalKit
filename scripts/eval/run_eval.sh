source .venv/bin/activate
set -ex

export HF_HUB_CACHE=/mnt/amlfs-02/shared/ckpts/mmc/

#  ChartQA_TEST DocVQA_VAL InfoVQA_VAL TextVQA_VAL AI2D_TEST MathVista_MINI
DATASET=${DATASET:-"A-OKVQA"}
CKPT=${CKPT:-"Qwen/Qwen2-VL-2B-Instruct"}

python -m debugpy_cli \
    run.py --data  \
    --model $CKPT --verbose --judge exact_matching

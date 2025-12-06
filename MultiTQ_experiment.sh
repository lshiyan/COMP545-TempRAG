#!/bin/bash
#SBATCH --output=logs/MultiTQ_experiment.log
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=def-sreddy
#module load python/3.10
#module load cuda/12.2

# (Optional) activate your virtual environment
#source .venv/bin/activate

# ==========================
# EXPERIMENT CONFIG
# ==========================
export CUDA_VISIBLE_DEVICES=0
QUESTIONS="/work/xinyu/COMP545-TempRAG/data/MultiTQ/full_questions.json"
INDEX="/work/xinyu/COMP545-TempRAG/data/MultiTQ/full_index.faiss"
METADATA="/work/xinyu/COMP545-TempRAG/data/MultiTQ/full_metadata.json"

SAMPLE_BY="qtype"      # ["any", "qtype", "qlabel", "answer_type", "time_level"]
SAMPLE_N=30       # sample size per group (ignored if SAMPLE_BY=any)

SIM_TOP_K=2000           # FAISS retrieval size
RERANK_TOP_K=50        # reranker selection

# ==========================
# RUN PROGRAM
# ==========================

LOG_FILE="logs/MultiTQ_experiment_${SAMPLE_BY}_gpu${CUDA_VISIBLE_DEVICES}_trial2.log"
echo "Logging to ${LOG_FILE}"

nohup python3 -m experiments.MultiTQ \
    --questions $QUESTIONS \
    --index $INDEX \
    --metadata $METADATA \
    --sample_by $SAMPLE_BY \
    --sample_n $SAMPLE_N \
    --similarity_top_k $SIM_TOP_K \
    --rerank_top_k $RERANK_TOP_K > "${LOG_FILE}" 2>&1 &

echo "Finished at: $(date)"

#!/bin/bash
#SBATCH --output=logs/MultiTQ_experiment.log
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=def-sreddy
module load python/3.10
module load cuda/12.2

# (Optional) activate your virtual environment
source .venv/bin/activate

# ==========================
# EXPERIMENT CONFIG
# ==========================

QUESTIONS="data/tkg/MultiTQ/full/full_questions.json"
INDEX="data/tkg/MultiTQ/full/full_index.faiss"
METADATA="data/tkg/MultiTQ/full/full_metadata.json"

SAMPLE_BY="any"      # ["any", "qtype", "qlabel", "answer_type", "time_level"]
SAMPLE_N=100             # sample size per group (ignored if SAMPLE_BY=any)

SIM_TOP_K=100           # FAISS retrieval size
RERANK_TOP_K=50        # reranker selection

# ==========================
# RUN PROGRAM
# ==========================

python3 -m experiments.MultiTQ \
    --questions $QUESTIONS \
    --index $INDEX \
    --metadata $METADATA \
    --sample_by $SAMPLE_BY \
    --sample_n $SAMPLE_N \
    --similarity_top_k $SIM_TOP_K \
    --rerank_top_k $RERANK_TOP_K

echo "Finished at: $(date)"


#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL

##SBATCH -o eapMT_tatoeba%j.out
#SBATCH -o attention_change_llama_coqa%j.out
##SBATCH -o layerwise_embedding_yelp%j.out

# Define job name
##SBATCH -J twitter


# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-lowbig
#SBTACH -N 1
#SBATCH -n 8
# Load the necessary modules

module load miniforge3/25.3.0-python3.12.10
source activate principle
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ((i=1; i<13; i++))
do
	python sort_edges.py $i 
done

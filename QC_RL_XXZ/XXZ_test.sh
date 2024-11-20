#!/bin/bash
#SBATCH --job-name=XXZ-test
#SBATCH --partition=CLUSTER
#SBATCH -t 7-00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=compute-0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1GB
#SBATCH -o ./output/%x-%j.out
#SBATCH -e ./output/%x-%j.err

module load python/anaconda3
module load cuda/cuda-11.4

source activate tf_tc

n_views="50"
target_case="1 2 3 4 5 6 7"

for nv in $n_views
do
  for tc in $target_case
    do
      python XXZ_test.py \
      --n-views $nv \
      --target-case $tc \
      --seed 42
  done
done
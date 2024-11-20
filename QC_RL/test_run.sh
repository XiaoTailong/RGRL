#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=CLUSTER
#SBATCH -t 7-00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=compute-0-2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1GB
#SBATCH -o ./output/%x-%j.out
#SBATCH -e ./output/%x-%j.err


module load python/anaconda3
module load cuda/cuda-11.4

source activate tf_tc

n_views="15"
action_step="0.05"
target_case="2"
agent_random="True"

for v in $n_views
  do
  for a in $action_step
    do
      for ar in $agent_random
      do
        for tc in $target_case
        do
          python Ising_QC_test.py \
          --n-views $v \
          --action-step $a \
          --num-steps 512 \
          --agent-random $ar \
          --num-minibatches 4 \
          --maximum-counter 512 \
          --target-case $tc \
          --seed 15
        done
      done
    done
  done
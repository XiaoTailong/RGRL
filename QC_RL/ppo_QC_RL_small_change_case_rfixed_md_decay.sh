#!/bin/bash
#SBATCH --job-name=PPO-md
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

action_step="0.1"
small_range="0.05 0.1"
target_case="1 2 3 4"
agent_random="True False"

for a in $action_step
  do
    for sr in $small_range
    do
      for ar in $agent_random
      do
        for tc in $target_case
        do
          python main_small_range_nodone_case_rfixed_multidiscrete_decay.py \
          --n-views 5 \
          --small-range $sr \
          --action-step $a \
          --num-steps 256 \
          --agent-random $ar \
          --learning-rate 6e-4 \
          --num-minibatches 4 \
          --target-case $tc \
          --seed 43 \
          --total-timesteps 50000 \
          --anneal-lr True \
          --max-grad-norm 0.5
        done
      done
    done
  done
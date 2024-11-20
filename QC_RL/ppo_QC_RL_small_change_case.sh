#!/bin/bash
#SBATCH --job-name=PPO
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

n_views="20 70"
action_step="0.05 0.04 0.03"
target_case="1 2 3 4"

for v in $n_views
  do
  for a in $action_step
    do
      for tc in $target_case
      do
        python main_small_range_nodone_case_v2.py \
        --n-views $v \
        --action-step $a \
        --num-steps 512 \
        --learning-rate 3e-4 \
        --num-minibatches 4 \
        --maximum-counter 512 \
        --target-case $tc \
        --seed 42 \
        --total-timesteps 100000 \
        --anneal-lr True \
        --max-grad-norm 0.5
      done
    done
  done
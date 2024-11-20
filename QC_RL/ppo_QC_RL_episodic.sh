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

n_views="70"
action_step="0.05"

for v in $n_views
  do
  for a in $action_step
    do
      python ppo_episodic.py \
      --n-views $v \
      --action-step $a \
      --seed 42 \
      --maximum-counter 100 \
      --total-timesteps 10000 \
      --anneal-lr False \
      --max-grad-norm 0.5
    done
  done
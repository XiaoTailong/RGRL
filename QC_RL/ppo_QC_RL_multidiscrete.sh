#!/bin/bash
#SBATCH --job-name=PPO_md
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

n_views="70"
action_step="0.02"

for v in $n_views
do
  for a in $action_step
    do
      python main_vec_multidiscrete.py \
      --n-views $v \
      --action-step $a \
      --num-steps 512 \
      --num-minibatches 8 \
      --maximum-counter 512 \
      --seed 42 \
      --total-timesteps 100000 \
      --anneal-lr False \
      --max-grad-norm 0.5
    done
done
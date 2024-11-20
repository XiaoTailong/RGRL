#!/bin/bash
#SBATCH --job-name=XXZ
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

n_views="100 200 70 50"
num_steps="256 512"


for v in $n_views
  do
  for ns in $num_steps
    do
      python main_small_change.py \
      --n-views $v \
      --num-steps $ns \
      --num-minibatches 8 \
      --maximum-counter 128 \
      --seed 42 \
      --total-timesteps 100000 \
      --anneal-lr True \
      --max-grad-norm 0.5 \
      --learning-rate 3e-4
    done
  done
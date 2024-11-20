#!/bin/bash
#SBATCH --job-name=PPO_vec
#SBATCH --partition=CLUSTER
#SBATCH -t 7-00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=compute-0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2GB
#SBATCH -o ./output/%x-%j.out
#SBATCH -e ./output/%x-%j.err


module load python/anaconda3
module load cuda/cuda-11.4

source activate tf_tc

n_views="70"
action_step="0.05"
target_index="2 3 4"
seed="42 43 44 45 46 47 48 50 51 52"


for v in $n_views
  do
  for a in $action_step
    do
      for ti in $target_index
      do
        for s in $seed
        do
          python main_vec_one_stage_two_actions.py \
          --n-views $v \
          --n-views-target 729 \
          --action-step $a \
          --num-steps 512 \
          --num-minibatches 4 \
          --target-index $ti \
          --num-envs 1 \
          --seed $s \
          --learning-rate 3e-4 \
          --total-timesteps 50000 \
          --update-epochs 4 \
          --anneal-lr True \
          --max-grad-norm 0.5
        done
      done
    done
  done
#!/bin/bash
#SBATCH --job-name=PPO_vec
#SBATCH --partition=CLUSTER
#SBATCH -t 7-00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=compute-0-2
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
action_step_2="0.01"
n_views_target="729"
target_index="0 1 2 3 4 5 6 7 8 9"

for v in $n_views
  do
  for a in $action_step
    do
      for a_2 in $action_step_2
      do
        for as in $n_views_target
        do
          for t_index in $target_index
          do
            python main_vec_two_stage.py \
            --n-views $v \
            --n-views-target $as \
            --action-step $a \
            --action-step-2 $a_2 \
            --reward-thr 1 \
            --accum-thr 30 \
            --target-index $t_index \
            --num-steps 1024 \
            --num-minibatches 8 \
            --num-envs 1 \
            --seed 42 \
            --learning-rate 3e-4 \
            --total-timesteps 100000 \
            --update-epochs 4 \
            --anneal-lr True \
            --max-grad-norm 0.5
          done
        done
      done
    done
  done
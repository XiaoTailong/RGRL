#!/bin/bash
#SBATCH --job-name=PPO-cgate
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

n_views="3"

## 改进空间为，因为每个action都存在不同的取值范围，那么每个action的步长，
## 需要设定不同decay，第一个参数从0.1开始decay，第二个参数从0.3开始decay，第三个参数从0.5开始decay

#先不decay，直接采用默认0.1的步长调控
for v in $n_views
do
  python main_cgate_discrete_three.py \
  --n-views $v \
  --action-step 0.005 \
  --num-steps 512 \
  --learning-rate 4e-4 \
  --num-minibatches 4 \
  --seed 36 \
  --total-timesteps 50000 \
  --anneal-lr True \
  --max-grad-norm 0.5
done

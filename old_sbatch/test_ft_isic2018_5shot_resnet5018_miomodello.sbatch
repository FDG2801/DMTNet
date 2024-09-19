#!/bin/bash

#SBATCH --job-name=<DMTNet_ISIC2017>
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=python-ex
#SBATCH --mail-type=ALL
#SBATCH --partition=cuda

###### 1 Load the module
module load nvidia/cudasdk/11.6
######module load intel/python/3/2019.4.088
# 2 Activate the virtual environment
#source thesis/bin/activate
# CONDA
source ~/.bashrc
conda activate condapip
# 3 Run the python script
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --node_rank=0 test_finetuning.py --backbone resnet50 --benchmark isic --nshot 5 --load pretrained_ckpt/resnet50_isic2018_fulltrained.pth --lr 1e-6
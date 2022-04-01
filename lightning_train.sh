#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --gres=gpu:2
#SBATCH --mail-use=bacharya@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --time=48:00:00
#SBATCH --output=hy-brid-can.out
source /media/compute/homes/bacharya/miniconda3/bin/activate
conda activate tf-gpu 
export PYTHONPATH=${PYTHONPATH}:/media/compute/homes/bacharya/MTTS-CAN/
srun python3 /media/compute/homes/bacharya/MTTS-CAN/code/train.py -exp hy-can -i /media/compute/vol/rppg/cohface/features/ -o /media/compute/vol/rppg/cohface/checkpoints/ -temp Hybrid_CAN 

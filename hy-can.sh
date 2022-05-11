#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --qos=regular             # regular or long
#SBATCH --partition=volta_devel # volta_compute or volta_devel
#SBATCH --gres=gpu:8
#SBATCH --mail-use=bacharya@techfak.uni-bielefeld.de
#SBATCH --mail-type=START
#SBATCH --mail-type=END
#SBATCH --output=hy-brid-long.gpu
#SBATCH --time=0-04:00:00
#SBATCH --gpus=8
source /home/bacharya/MTTS-CAN/tf-gpu/bin/activate
export PYTHONPATH=${PYTHONPATH}:/home/bacharya/MTTS-CAN/
srun python3 /home/bacharya/MTTS-CAN/code/train.py -exp long-hy-can-gpu -i /work/data/bacharya/cohface/ -o /home/bacharya/cohface/checkpoints/ -temp Hybrid_CAN 

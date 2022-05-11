#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --qos=regular             # regular or long
#SBATCH --partition=volta_devel   # volta_compute or volta_devel
#SBATCH --mail-use=bacharya@techfak.uni-bielefeld.de
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --output=PURE-3-multistep.log
#SBATCH --time=0-00:30:00
#SBATCH --gpus=4
source /home/bacharya/MTTS-CAN/tf-gpu/bin/activate
export PYTHONPATH=${PYTHONPATH}:/home/bacharya/MTTS-CAN/
srun python3 /home/bacharya/MTTS-CAN/code/train.py -exp shortpure-hy-can-3-4gpu -i /home/bacharya/rPPG_Deeplearning/src/features/PURE/ -o /home/bacharya/PURE/checkpoints/ -temp Hybrid_CAN -m 1 -init 0

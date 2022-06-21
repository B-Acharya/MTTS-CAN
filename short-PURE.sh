#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --partition=volta_devel   # volta_compute or volta_devel
#SBATCH --mail-use=bacharya@techfak.uni-bielefeld.de
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --output=Cohface-TS-0-multistep.log
#SBATCH --time=0-00:30:00
#SBATCH --gpus=4
source /home/bacharya/MTTS-CAN/tf-gpu/bin/activate
export PYTHONPATH=${PYTHONPATH}:/home/bacharya/MTTS-CAN/
srun python3 /home/bacharya/MTTS-CAN/code/train-coh.py -exp short-test-val-gpu -i /work/data/bacharya/cohface/ -o /home/bacharya/cohface/checkpoints/ -temp Hybrid_CAN -m 1 -init 1

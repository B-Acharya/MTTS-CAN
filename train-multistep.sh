#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --partition=volta_devel# volta_compute or volta_devel
#SBATCH --mail-use=bacharya@techfak.uni-bielefeld.de
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=0-00:30:00
#SBATCH --gpus=$gpus
source /home/bacharya/MTTS-CAN/tf-gpu/bin/activate
export PYTHONPATH=${PYTHONPATH}:/home/bacharya/MTTS-CAN/
echo $exp_name
echo $data
echo $save_dir
echo $temporal
echo $initial
python3 /home/bacharya/MTTS-CAN/code/train-coh.py -exp $exp_name -i $data -o $save_dir -temp $temporal -m 1 -init $initial -inter $inter

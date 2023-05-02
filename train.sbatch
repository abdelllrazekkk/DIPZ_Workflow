#! /bin/bash

#SBATCH -J train-base       # Job name
#SBATCH -o output-base.log  # Output file
#SBATCH -p gpgpu-1          # Partition
#SBATCH --exclusive         # Exclusivity 
#SBATCH --mem=250G          # Total memory required
#SBATCH --gres=gpu:1        # GPUs required

module purge

eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/8000net

python script.py --insert_flag_here
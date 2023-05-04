#! /bin/bash

#SBATCH -J train-base2       # Job name
#SBATCH -o output-base-%j.log  # Output file
#SBATCH -p development      # Partition
#SBATCH --exclusive         # Exclusivity 
#SBATCH --mem=250G          # Total memory required
#SBATCH --gres=gpu:1        # GPUs required

module purge

eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/8000net

python script.py --model=large_dense_x2 -t -e
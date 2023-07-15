#!/bin/bash

#SBATCH -J run_notebook              # Job name to display in the queue
#SBATCH -o run_notebook_%j.txt       # terminal output from the job, %j expands to the job id
#SBATCH -e run_notebook_%j.err       # error log from the job, %j expands to the job id
#SBATCH -p run_notebook_%j.err
#SBATCH --partition=standard-mem-s   # partition to run on
#SBATCH --nodes=1                    # number of nodes to run on
#SBATCH --ntasks-per-node=1          # number of tasks per node
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --time=0-00:30:00            # job runtime, format is D-HH:MM:SS
#SBATCH --mem=10G                    # memory for the job, in GB
#SBATCH --gres=gpu:0                 # number of GPUs gpu:1 = 1 gpu, gpu2:2 = 2 gpu

# make sure there are no modules loaded from the environment
module purge

# load the modules we want. We just need the python module
# for jupyterlabs and/or Anacaonda. If you are using other
# modules add them here
module load python/3

# make sure Conda commands are available in the script
# and load our Conda environment. If you are not using
# a Conda environment, you can omit these.
eval "$(conda shell.bash hook)"
conda activate my_conda_environment

# run the notebook
# This command is usually used to convert a notebook to a python script,
# but we can also use it to run the notebook and write the output into
# the same notebook, so when you open it the output areas are populated
jupyter nbconvert --to notebook --inplace --execute my_jupyter_notebook.ipynb

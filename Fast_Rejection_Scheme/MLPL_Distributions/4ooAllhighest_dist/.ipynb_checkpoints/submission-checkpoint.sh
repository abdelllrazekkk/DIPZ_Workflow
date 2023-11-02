#!/bin/bash

#SBATCH -J 4ooAllhighest_dist       # job name to display in squeue
#SBATCH -o output-4ooAllhighest_dist.txt    # standard output file
#SBATCH -e error-4ooAllhighest_dist.txt     # standard error file
#SBATCH -p htc      # requested partition
#SBATCH --mem=500G          # Total memory required
#SBATCH -t 1400              # maximum runtime in minutes
#SBATCH -D /users/maboelela/Research/Qualification_Task/DIPZ_Workflow/4ooAllhighest_dist  #sets the working directory where the batch script should be run
#SBATCH -s   #tells SLURM that the job can not share nodes with other running jobs
#SBATCH --mail-user maboelela@smu.edu   #tells SLURM your email address if you’d like to receive job-related email notifications
#SBATCH --mail-type=all

module purge

eval "$(conda shell.bash hook)"
conda activate /lustre/work/client/users/maboelela/.conda/envs/dipz

time python run_and_save.py -t -e
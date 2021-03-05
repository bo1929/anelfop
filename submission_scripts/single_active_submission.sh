#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=te_991_cl4l_pca200_s800
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=2
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=23:59:00
#SBATCH --output=/cta/users/aosman/AL4NER/submission_scripts/s800_al/te_991_cl4l_pca200_s800.out
#SBATCH --mem-per-cpu=32G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python /cta/users/aosman/AL4NER/al_experiment.py /cta/users/aosman/AL4NER/config_files/s800/te_991_cl4l_pca200_s800_active_experiment_config.yaml


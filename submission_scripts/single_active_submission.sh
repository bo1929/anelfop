#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=nap_cl4l_pca200_NCBI_disease
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=8
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=23:59:00
#SBATCH --output=/cta/users/aosman/alNER/submission_scripts/NCBI_disease_al/nap_cl4l_pca200_NCBI_disease.out
#SBATCH --mem-per-cpu=8G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python /cta/users/aosman/alNER/al_experiment.py /cta/users/aosman/alNER/config_files/NCBI_disease/nap_cl4l_pca200_NCBI_disease_active_exp_config.yaml


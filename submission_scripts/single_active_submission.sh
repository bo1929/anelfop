#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=nap_ll_pca300_debug_CONLL2003
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=4
#SBATCH --qos=short_mdbf
#SBATCH --partition=short_mdbf
#SBATCH --time=1:59:00
#SBATCH --output=/cta/users/aosman/alNER/submission_scripts/debug_CONLL2003_al/nap_ll_pca300_debug_CONLL2003.out
#SBATCH --mem-per-cpu=1G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python /cta/users/aosman/alNER/al_experiment.py /cta/users/aosman/alNER/config_files/debug_CONLL2003/nap_ll_pca300_debug_CONLL2003_active_exp_config.yaml


#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=sl4l_pca300_debug_CONLL2003
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=4
#SBATCH --qos=short_mdbf
#SBATCH --partition=short_mdbf
#SBATCH --time=1:59:00
#SBATCH --output=/cta/users/aosman/alNER/submission_scripts/debug_CONLL2003_pl/sl4l_pca300_debug_CONLL2003.out
#SBATCH --mem-per-cpu=1G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python /cta/users/aosman/alNER/pl_experiment.py /cta/users/aosman/alNER/config_files/debug_CONLL2003/sl4l_pca300_debug_CONLL2003_passive_exp_config.yaml


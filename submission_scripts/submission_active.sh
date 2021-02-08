#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
echo ${curr_dir}

rmpost="submission_scripts/"
job_dir=${curr_dir%"submission_scripts/"}
echo ${job_dir}

method_all=("ptp" "ptm" "pte" "pap" "rs" "lss" "tp" "ttp" "ntp" "tm" "ttm" "ntm" "te" "tte" "nte" "ap" "tap" "nap")
seed_all=(121 211 112 919 991 199 722 227 272)
data="debug_CONLL2003"
embedding_type="cl4l"
reduction="pca256" #off

increment_size="exp1"
init_size="16"
stopping_criteria="full" #full

if [ -d ${job_dir}"config_files/" ]; then
    echo "File exists"
else
    mkdir ${job_dir}"config_files/"
fi

if [ -d ${job_dir}"config_files/${data}/" ]; then
    echo "File exists"
else
    mkdir ${job_dir}"config_files/${data}/"
fi

if [ -d ${curr_dir}"${data}_al/" ]; then
    echo "File exists"
else
    mkdir ${curr_dir}"${data}_al/"
fi

for method_ in ${method_all[@]}; do
    for seed_ in ${seed_all[@]}; do
        name="${method_}_${seed_}_${embedding_type}_${reduction}_${data}"
        config_path="${job_dir}config_files/${data}/${name}_active_experiment_config.yaml"


        echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=8
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=23:59:00
#SBATCH --output=${curr_dir}${data}_al/${name}.out
#SBATCH --mem-per-cpu=8G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python ${job_dir}al_experiment.py ${config_path}
"       >${curr_dir}"single_active_submission.sh"


        ${curr_dir}"wrt_active_conf.sh" -d ${data} -m ${method_} -e ${embedding_type} -r ${reduction} -c ${increment_size} -s ${stopping_criteria} -i ${init_size} -p ${job_dir} --seed ${seed_} >${config_path}
        sbatch ${curr_dir}"single_active_submission.sh"
    done
done

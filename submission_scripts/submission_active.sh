#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
echo ${curr_dir}

rmpost="submission_scripts/"
job_dir=${curr_dir%"submission_scripts/"}
echo ${job_dir}

method_all=("ptp" "ptm" "pte" "pap" "rs" "lss" "tp" "ttp" "ntp" "tm" "ttm" "ntm" "te" "tte" "nte" "ap" "tap" "nap")

data="s800"
embedding_type="cl4l"
reduction="pca200" #off

increment_size="cp3"
init_size="p2"
stopping_criteria="ge50"

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

for mthd in ${method_all[@]}; do
    name="${mthd}_${embedding_type}_${reduction}_${data}"
    config_path="${job_dir}config_files/${data}/${name}_active_exp_config.yaml"
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
    " >${curr_dir}"single_active_submission.sh"
    ${curr_dir}"wrt_active_conf.sh" -d ${data} -m ${mthd} -e ${embedding_type} -r ${reduction} -c ${increment_size} -s ${increment_size} -i ${init_size} -p ${job_dir} >${config_path}
    sbatch ${curr_dir}"single_active_submission.sh"
done

#!/bin/bash

curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${curr_dir}
rmpost="submission_scripts"
job_dir=${curr_dir%"submission_scripts"}
echo ${job_dir}

method_all=("rs" "lss" "tp" "ttp" "ntp" "tm" "ttm" "ntm" "te" "tte" "nte" "ap" "tap" "nap" "ptp" "ptm" "pte" "pap")

data="CONLL2003"
increment_size="p0.5"
init_size="p0.5"

if [ -d ${job_dir}"/config_files/" ]; then
    echo "File exists"
else  
    mkdir ${job_dir}"/config_files/"
fi 

if [ -d ${job_dir}"/config_files/${data}/" ]; then
    echo "File exists"
else  
    mkdir ${job_dir}"/config_files/${data}/"
fi 

if [ -d ${curr_dir}"/${data}/" ]; then
    echo "File exists"
else  
    mkdir ${curr_dir}"/${data}/"
fi 

for mthd  in ${method_all[@]}; do

    name="${mthd}_${data}"
    config_path="${job_dir}/config_files/${data}/${name}_active_exp_config.yaml"

    echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=8
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=11:59:00
#SBATCH --output=${curr_dir}/${data}/${name}.out
#SBATCH --mem-per-cpu=8G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python ${job_dir}al_experiment.py ${config_path}
" > ${curr_dir}"/single_submission.sh"
    ${curr_dir}"/wrt_active_conf.sh" -d $data -m $mthd -c $increment_size -i $init_size -r ${job_dir} > ${config_path}
    sbatch ${curr_dir}"/single_submission.sh"
done

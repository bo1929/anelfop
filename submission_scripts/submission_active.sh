#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
echo ${curr_dir}

rmpost="submission_scripts/"
job_dir=${curr_dir%"submission_scripts/"}
echo ${job_dir}

method_all=("ttp")
seed_all=(211)
data="CONLL2003"
embedding_type="cl4l"
reduction="pca200" #off

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

        if [ ${method_} = "ptp" ] || [ ${method_} = "ptm" ] || [ ${method_} = "pte" ] || [ ${method_} = "pap" ];
        then
          cpu="8"
          mem_per_cpu="8G"
        else
          cpu="2"
          mem_per_cpu="32G"
        fi

        echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=${cpu}
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=23:59:00
#SBATCH --output=${curr_dir}${data}_al/${name}.out
#SBATCH --mem-per-cpu=${mem_per_cpu}

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

#!/bin/bash

curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/"
echo ${curr_dir}
rmpost="submission_scripts/"
job_dir=${curr_dir%"submission_scripts/"}
echo ${job_dir}

data="CONLL2003"
embedding_type_all=("ll" "cl4l" "sl4l")
reduction_all=("off" "pca200" "pca256" "pca300")

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

if [ -d ${curr_dir}"${data}_pl/" ]; then
    echo "File exists"
else  
    mkdir ${curr_dir}"${data}_pl/"
fi 

for embedding_type  in ${embedding_type_all[@]}; do
    for reduction in ${reduction_all[@]}; do
        name="${embedding_type}_${reduction}_${data}"
        config_path="${job_dir}config_files/${data}/${name}_passive_exp_config.yaml"
        
        
        echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=8
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=23:59:00
#SBATCH --output=${curr_dir}${data}_pl/${name}.out
#SBATCH --mem-per-cpu=8G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python ${job_dir}pl_experiment.py ${config_path}
"> ${curr_dir}"single_passive_submission.sh"
        
        
        ${curr_dir}"wrt_passive_conf.sh" -d ${data} -e ${embedding_type} -r ${reduction} -p ${job_dir} > ${config_path}
        sbatch ${curr_dir}"single_passive_submission.sh"
    done
done

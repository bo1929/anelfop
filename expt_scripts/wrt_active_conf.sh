#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -d|--data)
    dataset="$2"
    shift # past argument
    shift # past value
    ;;
    -e|--embedding)
    embedding_type="$2"
    shift
    shift
    ;;
    -r|--reduction)
    reduction="$2"
    shift
    shift
    ;;
    -m|--method)
    method="$2"
    shift
    shift
    ;;
    -c|--inc)
    increment="$2"
    shift
    shift
    ;;
    -i|--init)
    init_size="$2"
    shift
    shift #past value
    ;;
    -s|--stop)
    stopping_criteria="$2"
    shift
    shift #past value
    ;;
    --seed)
    seed="$2"
    shift
    shift #past value
    ;;
    -p|--path)
    main_dir="$2"
    shift
    shift
    ;;
    *) # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift
    ;;
esac
done

if [ ${dataset} = "CONLL2003" ];
then
  pos="True"
  pre_model="bert-base-cased"
elif [ ${dataset} = "BC5CDR" ] || [ ${dataset} = "BC2GM" ] || [ ${dataset} = "Genia4ER" ];
then
  pos="False"
  # pre_model="emilyalsentzer/Bio_ClinicalBERT"
  pre_model="dmis-lab/biobert-base-cased-v1.1"
elif [ ${dataset} = "NCBI_disease" ]
then
  pos="False"
  pre_model="dmis-lab/biobert-base-cased-v1.1"
  # pre_model="emilyalsentzer/Bio_ClinicalBERT"
elif [ ${dataset} = "s800" ] || [ ${dataset} = "LINNAEUS" ];
then
  pos="False"
  pre_model="dmis-lab/biobert-base-cased-v1.1"
  # pre_model="emilyalsentzer/Bio_ClinicalBERT"
elif [ ${dataset:0:5} = "debug" ];
then
  pos="False"
  pre_model="bert-base-cased"
else
  exit 128
fi


echo "seed: ${seed}

increment_cons: ${increment}
initial_size: ${init_size}
stopping_criteria: ${stopping_criteria}

generator: False
method: ${method}

main_directory: ${main_dir}
data_directory: ${main_dir}datasets/tokenized/

data_set:
  name: ${dataset}
  pos: ${pos}

pretrained_model: ${pre_model}

embedding_type: ${embedding_type}

init_reduction:
  type: ${reduction:0:3}
  pca:
    dimension: ${reduction:3}

CRF:
  algorithm: lbfgs
  c1: 0.1
  c2: 0.1
  max_iterations: 100
  allow_all_states: True
  allow_all_transitions: True

outlier_method:
  method_name: GLOSH
  outlier_quantile: 0.995
  parameters: {}

hdbscan_al:
  min_c_size: 1000
  min_samp: 70
  c_eps: 0.19

umap_al:
  neig: 40
  min_dist: 0.0
  n_comp: 2
"

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
    -m|--meth)
    method="$2"
    shift # past argument
    shift # past value
    ;;
    -c|--inc)
    increment="$2"
    shift # past argument
    shift # past value
    ;;
    -i|--init)
    init_size="$2"
    shift # past argument
    shift #past value
    ;;
    -r|--mdir)
    main_dir="$2"
    shift
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

embedding_type="ll"

if [ $data="CONLL2003" ]; 
then
  pos="True"
  pre_model="bert-base-cased"
else
  pos="False"
  pre_model=""
fi

echo "seed: 219

increment_cons: ${increment}
initial_size: ${init_size}

generator: True
method: ${method} 

main_directory: ${main_dir}
data_directory: ${main_dir}/datasets/tokenized/

data_set:
  name: ${dataset}
  pos: ${pos}

pretrained_model: ${pre_model}

#sl4l
#ll
#cl4l
embedding_type: ${embedding_type}

init_reduction:
  type: pca
  pca:
    dimension: 300
  umap:
    dimension: 300
    min_dist: 0.0
    neig: 40

CRF:
  algorithm: lbfgs
  c1: 0.1
  c2: 0.1
  max_iterations: 100
  allow_all_states: True
  allow_all_transitions: True

hdbscan_al:
  mask_outlier: 0.99
  min_c_size: 1000
  min_samp: 200
  c_eps: 0.17

umap_al:
  neig: 40
  min_dist: 0.0
  n_comp: 2
"

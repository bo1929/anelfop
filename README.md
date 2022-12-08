# ANELFOP
Active Named Entity Learning by Focusing on Possible Named Entities

Please do not forget to edit the config files. Also, if you do not want to use them from scratch, you can use the `expt_scripts` for ease of usage.

## For Reproducing

First create a virtual environment based on `python 3.7`.

Then install the required packages with `pip install requirements.txt`.

### For Active Learning
You can use `python anelfop/al_experiment.py`

You can use the following config file:
```yaml
seed: seed
increment_cons: increment
initial_size: init_size
stopping_criteria: stopping_criteria
generator: False
method: method
main_directory: main_dir
data_directory: main_dir/datasets/tokenized/
data_set:
  name: dataset
  pos: pos
pretrained_model: pre_model
embedding_type: embedding_type
init_reduction:
  type: reduction:0:3
  pca:
    dimension: reduction:3
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
```

### For Passive Learning
You can use `python anelfop/pl_experiment.py`

You can use the following config file:
```yaml
seed: 219
generator: True
method: method
main_directory: main_dir
data_directory: main_dir/datasets/tokenized/
data_set:
  name: dataset
  pos: pos
pretrained_model: pre_model
embedding_type: embedding_type
init_reduction:
  type: reduction:0:3
  pca:
    dimension: reduction:3
CRF:
  algorithm: lbfgs
  c1: 0.1
  c2: 0.1
  max_iterations: 100
  allow_all_states: True
  allow_all_transitions: True
```

### For Clustering
You can use `python anelfop/ss_clustering.py`

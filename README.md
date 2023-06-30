# Focusing on potential named entities during active label acquisition
Implementation of the method (`anelfop`) described in *Şapcı, A., Kemik, H., Yeniterzi, R., & Tastan, O. (2023). Focusing on potential named entities during active label acquisition. Natural Language Engineering, 1-23*[^1], together scripts used for experiments discussed in the paper.

[^1]: doi:10.1017/S1351324923000165

## About the method
Named entity recognition (NER) aims to identify mentions of named entities in an unstructured text and classify them into predefined named entity classes.
While deep learning-based pre-trained language models help to achieve good predictive performances in NER, many domain-specific NER applications still call for a substantial amount of labeled data.
Active learning (AL), a general framework for the label acquisition problem, has been used for NER tasks to minimize the annotation cost without sacrificing model performance.
However, the heavily imbalanced class distribution of tokens introduces challenges in designing effective AL querying methods for NER.
We propose several AL sentence query evaluation functions that pay more attention to potential positive tokens and evaluate these proposed functions with both sentence-based and token-based cost evaluation strategies.
We also propose a better data-driven normalization approach to penalize sentences that are too long or too short.
Our experiments on three datasets from different domains reveal that the proposed approach reduces the number of annotated tokens while achieving better or comparable prediction performance with conventional methods.

## Reproducing the results
If you would like to re-produce the experiments, see `expt_scripts`.
Do not forget to edit the config files.

First create a virtual environment based on `python 3.7`.

Then install the required packages with `pip install requirements.txt`.

### For active learning
You can use `python anelfop/al_experiment.py`, and the following configuration file:
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

### For passive learning
You can use `python anelfop/pl_experiment.py`, and the following configuration file:
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

import operator
import random
import argparse
import numpy as np
import pandas as pd

from itertools import accumulate
from sklearn.metrics import classification_report

import functions
import load_save

import wrappers.wrapper_pretrained as pretrained_
import wrappers.wrapper_UMAP as umap_

parser = argparse.ArgumentParser(prog="PROG")
parser.add_argument(
    "--config-path",
    required=True,
    type=str,
    help="Configuration path to use, \
        seed, UMAP and HDBSCAN parameters",
)
parser.add_argument(
    "--labeled-percentage",
    required=True,
    type=int,
    help="What percentage of tokens \
                will have  for semi-supervised \
                dimensionality reduction.",
)

args = parser.parse_args()
config_path = args.config_path
labeled_percentage = args.labeled_percentage

cfg = load_save.load_config_from(config_path)
random_seed = cfg["seed"]
kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

[tknzd_sent_train, tags_train, pos_train], [
    tknzd_sent_test,
    tags_test,
    pos_test,
] = load_save.load_data(cfg)
tag_dict = cfg["tag_dict"]

(
    embeddings_train,
    pretrained_tknzd_train,
    tknzd_sent_train,
    y_train,
    pos_train,
) = pretrained_.get_embeddings(
    cfg, tknzd_sent_train, tags_train, pos_train, part="train"
)

embeddings_train, _ = functions.reduce_embeddings(cfg, embeddings_train, [[]])

labeled_percentage = 0
num_sentences = len(embeddings_train)
idx_labeled = random.sample(
    [i for i in range(num_sentences)], num_sentences // 100 * labeled_percentage
)
idx_unlabeled = [i for i in range(num_sentences) if i not in idx_labeled]

embeddings_ann = [embeddings_train[i] for i in idx_labeled]
y_ann = [y_train[i] for i in idx_labeled]
embeddings_pool = [embeddings_train[i] for i in idx_unlabeled]
y_pool = [y_train[i] for i in idx_unlabeled]

(
    embeddings_ann,
    embeddings_pool,
    clusters_ann,
    clusters_pool,
    clusterer,
    count_clusters,
) = umap_.ss_umap_r_hdbscan_c(
    embeddings_ann, embeddings_pool, y_ann, tag_dict, seed=cfg["seed"], **kwargs
)

embeddings_ann_flatten = []
c_ann_flatten = []
for i in range(len(embeddings_pool)):
    embeddings_ann_flatten.append(embeddings_pool[i].reshape(-1, 2))
for i in range(len(clusters_pool)):
    c_ann_flatten.append(clusters_pool[i].flatten())

sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
sent_idx_pool = list(accumulate(sent_len_pool))

sent_len_ann = [0] + [len(sent) for sent in embeddings_ann]
sent_idx_ann = list(accumulate(sent_len_ann))


n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
threshold = pd.Series(clusterer.outlier_scores_[:]).quantile(
    cfg["hdbscan_al"]["mask_outlier"]
)
outliers = np.where(clusterer.outlier_scores_[:] > threshold)[0]
mask_out = np.zeros(len(clusterer.outlier_scores_[:]))
mask_out[outliers] = 1
mask_out_pool = mask_out[len(embeddings_ann) :]
mask_out_pool = [
    mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
    for i in range(1, len(sent_idx_pool))
]
mask_out_ann = mask_out[: len(embeddings_ann)]
mask_out_ann = [
    mask_out[sent_idx_ann[i - 1] : sent_idx_ann[i]] for i in range(1, len(sent_idx_ann))
]

y_true = []
y_pred = []

for i in range(len(embeddings_ann)):
    len_sent = len(embeddings_ann[i])
    for j in range(len_sent):
        if clusters_ann[i][j] != n_ent or mask_out_ann[i][j] == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
        if y_ann[i][j] == "O":
            y_true.append(0)
        else:
            y_true.append(1)

for i in range(len(embeddings_pool)):
    len_sent = len(embeddings_pool[i])
    for j in range(len_sent):
        if clusters_pool[i][j] != n_ent  or mask_out_pool[i][j] == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
        if y_pool[i][j] == "O":
            y_true.append(0)
        else:
            y_true.append(1)

report = classification_report(y_true, y_pred)
print(report)

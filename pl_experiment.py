import torch
import tqdm
import json
import random
import time as money

import os

import numpy as np

import utils.wrapper_CRF as crf_
import utils.wrapper_pretrained as pretrained_
import utils.wrapper_UMAP as umap_
import functions

import load_save
from operator import itemgetter
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report, f1_score

path_config = sys.argv[1] 

cfg = load_save.load_config_from(path_config, AL=False)
[tknzd_sent_train, tags_train, pos_train], [
    tknzd_sent_test,
    tags_test,
    pos_test,
] = load_save.load_data(cfg)

random_seed = cfg["seed"]

(
    embeddings_train,
    pretrained_tknzd_train,
    tknzd_sent_train,
    y_train,
    pos_train,
) = pretrained_.get_embeddings(
    cfg, tknzd_sent_train, tags_train, pos_train, part="train"
)

(
    embeddings_test,
    pretrained_tknzd_test,
    tknzd_sent_test,
    y_test,
    pos_test,
) = pretrained_.get_embeddings(cfg, tknzd_sent_test, tags_test, pos_test, part="test")


embeddings_train_r, embeddings_test_r = functions.reduce_embeddings(
    cfg,
    embeddings_train,
    embeddings_test,
)

embedding_dim = embeddings_train_r[0][0].shape[0]
print("final embedding dimension: ", embedding_dim)

load_save.write_ft_config(cfg)
feature_cfg = load_save.load_ft_config(cfg)

X_test = crf_.sent2features(
    feature_cfg,
    tknzd_sent_test,
    generator=cfg["generator"],
    embeddings=embeddings_test_r,
    pos=pos_test,
)

X_train = crf_.sent2features(
    feature_cfg,
    tknzd_sent_train,
    generator=cfg["generator"],
    embeddings=embeddings_train_r,
    pos=pos_train,
)

start = money.time()
print("Initial training CRF with annotated sentences...\n")
crf_trained = crf_.train_crf(cfg, X_train, y_train)

print("CRF test sentences predictions...\n")
y_pred = crf_trained.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
end = money.time()

load_save.save_crf_model(cfg, crf_trained, 0)
load_save.save_results(
    cfg,
    [report, start - end, f1_score(y_test, y_pred)],
    [f1_score(y_test, y_pred)],
    [],
    [],
)

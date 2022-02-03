import random
import argparse
import time as money

import numpy as np
import wrappers.wrapper_CRF as crf_
import wrappers.wrapper_pretrained as pretrained_

import functions
import load_save

from seqeval.metrics import classification_report, f1_score

parser = argparse.ArgumentParser(prog="PROG")
parser.add_argument(
    "--config-path",
    required=True,
    type=str,
    help="Configuration file path to use, \
        seed, UMAP and HDBSCAN parameters",
)

args = parser.parse_args()
config_path = args.config_path

cfg = load_save.load_config_from(config_path)
random_seed = cfg["seed"]

[tknzd_sent_train, tags_train, pos_train], [
    tknzd_sent_test,
    tags_test,
    pos_test,
] = load_save.load_data(cfg)

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
    cfg, embeddings_train, embeddings_test
)

embedding_dim = embeddings_train_r[0][0].shape[0]

load_save.write_ft_config(cfg)
feature_cfg = load_save.load_ft_config(cfg)

initial_size = functions.get_init_size(cfg, len(y_train))

random.seed(a=random_seed, version=2)
initial_idx_ann = random.sample(
    [i for i in range(len(embeddings_train_r))], initial_size
)

idx_pool = list(
    np.setdiff1d(
        np.arange(0, len(embeddings_train_r)), initial_idx_ann, assume_unique=True
    )
)

idx_ann = initial_idx_ann

X_test = crf_.sent2features(
    feature_cfg,
    tknzd_sent_test,
    generator=cfg["generator"],
    embeddings=embeddings_test_r,
    pos=pos_test,
)

sents_ann_i = [tknzd_sent_train[x] for x in idx_ann]
embeddings_ann_i = [embeddings_train_r[x] for x in idx_ann]
y_ann = [y_train[x] for x in idx_ann]
pos_ann_i = [pos_train[x] for x in idx_ann]

Xi_train = crf_.sent2features(
    feature_cfg,
    sents_ann_i,
    generator=cfg["generator"],
    embeddings=embeddings_ann_i,
    pos=pos_ann_i,
)

start = money.time()
start_training = money.time()
print("Initial training...\n")
crf_trained = crf_.train_crf(cfg, Xi_train, y_ann)
end_training = money.time()
print(f"Elapsed training time is {end_training - start_training}.")

print("Initial testing...\n")
yi_pred = crf_trained.predict(X_test)
report = classification_report(y_test, yi_pred)
print(report)

stats_queries = []
f1_scores = []
queried_indexes = []
queried_sent_len = []

end = money.time()

stats_queries.append((report, start - end))
f1_scores.append(f1_score(y_test, yi_pred))
queried_indexes.append(initial_idx_ann)
queried_sent_len.append(
    [len(sent) for sent in [tknzd_sent_train[x] for x in initial_idx_ann]]
)

active_learner = cfg["method"]

iteration = 1
stop_condition = functions.stopping_criteria(
    cfg, iteration, len(idx_pool), len(embeddings_train_r), []
)

while not (stop_condition):
    kwargs = {}

    start = money.time()
    print(f"Iteration {iteration} is running...\n")

    # Feature dictionaries of each word, sentence by sentence.
    sents_pool_i = [tknzd_sent_train[x] for x in idx_pool]
    embeddings_pool_i = [embeddings_train_r[x] for x in idx_pool]
    pos_pool_i = [pos_train[x] for x in idx_pool]

    Xi_pool = crf_.sent2features(
        feature_cfg,
        sents_pool_i,
        generator=cfg["generator"],
        embeddings=embeddings_pool_i,
        pos=pos_pool_i,
    )

    if cfg["generator"] and active_learner in ["ap", "tap", "nap", "pap"]:
        Xi_pool_ = crf_.sent2features(
            feature_cfg,
            sents_pool_i,
            generator=cfg["generator"],
            embeddings=embeddings_pool_i,
            pos=pos_pool_i,
        )
        kwargs.update({"Xi_pool_2nd": Xi_pool_})

    crf_tagger = crf_trained.tagger_

    idx_q, idx_pool = functions.query(
        cfg,
        crf_trained,
        iteration,
        idx_pool,
        idx_ann,
        Xi_pool,
        embeddings_train,
        y_train,
        **kwargs,
    )
    idx_ann = idx_ann + idx_q

    sents_ann_i = [tknzd_sent_train[x] for x in idx_ann]
    embeddings_ann_i = [embeddings_train_r[x] for x in idx_ann]
    pos_ann_i = [pos_train[x] for x in idx_ann]
    y_ann = [y_train[x] for x in idx_ann]

    Xi_train = crf_.sent2features(
        feature_cfg,
        sents_ann_i,
        generator=cfg["generator"],
        embeddings=embeddings_ann_i,
        pos=pos_ann_i,
    )

    print("Training...\n")
    start_training = money.time()
    crf_trained = crf_.train_crf(cfg, Xi_train, y_ann)
    end_training = money.time()
    print(f"Elapsed training time is {end_training - start_training}.")

    if cfg["generator"]:
        X_test_ = crf_.sent2features(
            feature_cfg,
            tknzd_sent_test,
            generator=cfg["generator"],
            embeddings=embeddings_test_r,
            pos=pos_test,
        )
    else:
        X_test_ = X_test

    print("Testing...\n")
    yi_pred = crf_trained.predict(X_test_)
    report = classification_report(y_test, yi_pred)
    print(report)

    end = money.time()
    stats_queries.append((report, end - start))
    f1_scores.append(f1_score(y_test, yi_pred))
    queried_indexes.append(idx_q)
    queried_sent_len.append(
        [len(sent) for sent in [tknzd_sent_train[x] for x in idx_q]]
    )

    stop_condition = functions.stopping_criteria(
        cfg, iteration, len(idx_pool), len(embeddings_train_r), f1_scores[-1]
    )

    load_save.save_crf_model(cfg, crf_trained, iteration)

    iteration = iteration + 1

load_save.save_results(cfg, stats_queries, f1_scores, queried_indexes, queried_sent_len)

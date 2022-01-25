import os
import math
import random
import operator

import numpy as np
import pandas as pd

import wrappers.wrapper_UMAP as umap_

import matplotlib.pyplot as plt
import statsmodels.api as sm

from itertools import accumulate


def fit_distribution(lengths):
    dens = sm.nonparametric.KDEUnivariate(lengths)
    dens.fit()

    return dens


def lenght_prob(dens, len_sent):
    # prob = dens.evaluate(len_sent)
    prob = math.sqrt(dens.evaluate(len_sent))

    return prob


def rs(idx_pool, batch_size, seed):
    random.seed(a=seed, version=2)

    idx_q = random.sample(idx_pool, batch_size)
    return idx_q, [s for s in idx_pool if s not in idx_q]


def lss(sent_lenghts, idx_pool, batch_size):
    batch = np.argpartition(np.array(sent_lenghts) * (-1), batch_size - 1)[
        :batch_size
    ].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def tp(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    tp = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        tp[i] = 1 - min([(max(m_pool[i][j].values())) for j in range(len_sent)])

    batch = np.argpartition(tp * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def ttp(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    ttp = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        ttp[i] = sum([1 - (max(m_pool[i][j].values())) for j in range(len_sent)])

    batch = np.argpartition(ttp * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def ntp(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    ntp = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        ntp[i] = (
            sum([1 - (max(m_pool[i][j].values())) for j in range(len_sent)]) / len_sent
        )

    batch = np.argpartition(ntp * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Minumum Token Margin
def tm(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    tm = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        tm[i] = 1 - min(
            [
                max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2]
                for j in range(len_sent)
            ]
        )

    batch = np.argpartition(tm * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def ttm(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    ttm = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        ttm[i] = sum(
            [
                1 - (max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2])
                for j in range(len_sent)
            ]
        )
    batch = np.argpartition(ttm * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [s for s in idx_pool if s not in idx_q]


def ntm(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    ntm = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        ntm[i] = (
            sum(
                [
                    1 - (max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2])
                    for j in range(len_sent)
                ]
            )
            / len_sent
        )

    batch = np.argpartition(ntm * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [s for s in idx_pool if s not in idx_q]


def te(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    te = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        te[i] = max(
            [
                (-1) * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                for j in range(len_sent)
            ]
        )

    batch = np.argpartition(te * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [s for s in idx_pool if s not in idx_q]


def tte(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    tte = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        tte[i] = sum(
            [
                (-1) * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                for j in range(len_sent)
            ]
        )

    batch = np.argpartition(tte * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def nte(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    nte = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        nte[i] = (
            sum(
                [
                    (-1)
                    * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                    for j in range(len_sent)
                ]
            )
            / len_sent
        )

    batch = np.argpartition(nte * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def ap(m_pool, y_pred, idx_pool, batch_size):
    num_sent = len(m_pool)
    ap = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        ap[i] = 1 - min([(m_pool[i][j][y_pred[i][j]]) for j in range(len_sent)])

    batch = np.argpartition(ap * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def tap(m_pool, y_pred, idx_pool, batch_size):
    num_sent = len(m_pool)
    tap = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        tap[i] = sum([1 - m_pool[i][j][y_pred[i][j]] for j in range(len_sent)])

    batch = np.argpartition(tap * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def nap(m_pool, y_pred, idx_pool, batch_size):
    num_sent = len(m_pool)
    nap = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        nap[i] = (
            sum([1 - m_pool[i][j][y_pred[i][j]] for j in range(len_sent)]) / len_sent
        )

    batch = np.argpartition(nap * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def ptp(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size):
    PDF = fit_distribution([len(sent) for sent in embeddings_pool])

    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = (
            sum(
                [
                    1 - max(m_pool[i][j].values())
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
                ]
            )
        ) * lenght_prob(PDF, len_sent)

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def otp(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size):
    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = sum(
            [
                1 - max(m_pool[i][j].values())
                for j in range(len_sent)
                if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
            ]
        )

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def otm(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size):
    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = sum(
            [
                1 - max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2]
                for j in range(len_sent)
                if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
            ]
        )

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def ptm(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size):
    PDF = fit_distribution([len(sent) for sent in embeddings_pool])

    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = (
            sum(
                [
                    1 - max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2]
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
                ]
            )
        ) * lenght_prob(PDF, len_sent)

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def pte(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size):
    PDF = fit_distribution([len(sent) for sent in embeddings_pool])

    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled.",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = (
            sum(
                [
                    (-1)
                    * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
                ]
            )
        ) * lenght_prob(PDF, len_sent)

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def ote(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size):
    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled.",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = sum(
            [
                (-1) * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                for j in range(len_sent)
                if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
            ]
        )

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def pap(
    cfg, embeddings_ann, embeddings_pool, y_ann, y_pred, m_pool, idx_pool, batch_size
):
    PDF = fit_distribution([len(sent) for sent in embeddings_pool])

    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled.",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = (
            sum(
                [
                    1 - m_pool[i][j][y_pred[i][j]]
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
                ]
            )
        ) * lenght_prob(PDF, len_sent)

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def oap(
    cfg, embeddings_ann, embeddings_pool, y_ann, y_pred, m_pool, idx_pool, batch_size
):
    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled.",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = sum(
            [
                1 - m_pool[i][j][y_pred[i][j]]
                for j in range(len_sent)
                if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
            ]
        )

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


def pas(cfg, embeddings_ann, embeddings_pool, y_ann, idx_pool, batch_size):
    PDF = fit_distribution([len(sent) for sent in embeddings_pool])

    experiment_dir = cfg["experiment_directory"]
    tag_dict = cfg["tag_dict"]
    kwargs = {**cfg["umap_al"], **cfg["hdbscan_al"]}

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

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_idx_pool = list(accumulate(sent_len_pool))

    fig, ax = plt.subplots(1, figsize=(18, 10))

    clr = [c for sent in clusters_pool for c in sent]
    coord = np.array([xy for sent in embeddings_pool for xy in sent])
    ax.scatter(coord[:, 0], coord[:, 1], c=clr, s=0.4, cmap="Spectral", alpha=0.5)
    fig.suptitle(
        "Semi-supervised UMAP + HDBSCAN, " + str(len(y_ann)) + " sentences labeled.",
        fontsize=18,
    )
    plt.savefig(
        os.path.join(
            experiment_dir + "ss_umap_r_hdbscan_c_" + str(len(y_ann)) + ".png"
        ),
        dpi=700,
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann) :]).quantile(
        cfg["hdbscan_al"]["mask_outlier"]
    )
    outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann) :] > threshold)[0]
    mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann) :]))
    mask_out[outliers] = 1
    mask_out = [
        mask_out[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    num_sent = len(embeddings_pool)
    entity_rich = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        entity_rich[i] = (
            sum(
                [
                    1
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out[i][j] == 1)
                ]
            )
        ) * lenght_prob(PDF, len_sent)

    batch = np.argpartition(entity_rich * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]

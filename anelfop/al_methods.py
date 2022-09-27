import os
import math
import random
import operator

import numpy as np
import pandas as pd

import wrappers.wrapper_UMAP as umap_

import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.neighbors import LocalOutlierFactor

from itertools import accumulate


def fit_distribution(lengths):
    dens = sm.nonparametric.KDEUnivariate(lengths)
    dens.fit()
    return dens


def lenght_prob(dens, len_sent):
    # prob = dens.evaluate(len_sent)
    prob = math.sqrt(dens.evaluate(len_sent))
    return prob


# Random Selection: RS
def rs(idx_pool, batch_size, seed):
    random.seed(a=seed, version=2)

    idx_q = random.sample(idx_pool, batch_size)
    return idx_q, [s for s in idx_pool if s not in idx_q]


# Longest Sentence Selection: LSS
def lss(sent_lenghts, idx_pool, batch_size):
    batch = np.argpartition(np.array(sent_lenghts) * (-1), batch_size - 1)[
        :batch_size
    ].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Single Token Probability: sTP
def tp(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    tp = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        tp[i] = 1 - min([(max(m_pool[i][j].values())) for j in range(len_sent)])

    batch = np.argpartition(tp * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Token Probability: tTP
def ttp(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    ttp = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        ttp[i] = sum([1 - (max(m_pool[i][j].values())) for j in range(len_sent)])

    batch = np.argpartition(ttp * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Normalized Token Probability: nTP
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


# Single Token Margin: sTM
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


# Total Token Margin: tTM
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


# Normalized Token Margin: nTM
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


# Single Token Entropy: nTE
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


# Total Token Entropy: tTE
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


# Normalized Token Entropy: nTE
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


# Single Assignment Probability: sAP
def ap(m_pool, y_pred, idx_pool, batch_size):
    num_sent = len(m_pool)
    ap = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        ap[i] = 1 - min([(m_pool[i][j][y_pred[i][j]]) for j in range(len_sent)])

    batch = np.argpartition(ap * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Assignment Probability: tAP
def tap(m_pool, y_pred, idx_pool, batch_size):
    num_sent = len(m_pool)
    tap = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        tap[i] = sum([1 - m_pool[i][j][y_pred[i][j]] for j in range(len_sent)])

    batch = np.argpartition(tap * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Normalized Assignment Probability: nAP
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


# Density Normalized Positive Token Probability: dpTP
def ptp(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size, plot=True):
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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


def plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann):
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


def compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool, outlier_method="GLOSH"):
    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    if outlier_method == "GLOSH":
        threshold = pd.Series(clusterer.outlier_scores_[len(embeddings_ann):]).quantile(
            cfg["hdbscan_al"]["mask_outlier"]
        )
        outliers = np.where(clusterer.outlier_scores_[len(embeddings_ann):] > threshold)[0]
        mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann):]))
        mask_out[outliers] = 1
        mask_out = [
            mask_out[sent_idx_pool[i - 1]: sent_idx_pool[i]]
            for i in range(1, len(sent_idx_pool))
        ]
    elif outlier_method == "LOF":
        clf = LocalOutlierFactor(contamination='auto') # sets threshold to 0.1
        lof_outliers1 = clf.fit_predict(embeddings_ann)
        mask_out = [0 if outlier_score==1 in lof_outliers1 else 1 for outlier_score in lof_outliers1]
    elif outlier_method == None:
        mask_out = np.zeros(len(clusterer.outlier_scores_[len(embeddings_ann):]))
    else:
        raise Exception("An outlier method from the following list must be specified: ['GLOSH','LOF',None]")
    return mask_out, n_ent


# Total Positive Token Probability: dP
def otp(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size, plot=True):
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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


# Total Positive Token Margin: tpTM
def otm(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size, plot=True):
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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


# Densitiy Normalized Positive Token Margin: dpTM
def ptm(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size, plot=True):
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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


# Densitiy Normalized Positive Token Entropy: dpTE
def pte(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size, plot=True):
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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


# Total Positive Token Entropy: tpTE
def ote(cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size, plot=True):
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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


# Density Normalized Positive Assignment Probability: dpAP
def pap(
    cfg, embeddings_ann, embeddings_pool, y_ann, y_pred, m_pool, idx_pool, batch_size, plot=True
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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


# Total Positive Assignment Probability: tpAP
def oap(
    cfg, embeddings_ann, embeddings_pool, y_ann, y_pred, m_pool, idx_pool, batch_size, plot=True
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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


# Positive Annotation Selection: PAS
def pas(cfg, embeddings_ann, embeddings_pool, y_ann, idx_pool, batch_size, plot=True):
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

    if plot: plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out, n_ent = compute_outliers(cfg, clusterer, count_clusters, embeddings_ann, sent_idx_pool)

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

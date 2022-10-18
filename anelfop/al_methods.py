import os
import math
import random
import operator

import numpy as np

import wrappers.wrapper_UMAP as umap_

import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.neighbors import LocalOutlierFactor

from itertools import accumulate


def fit_distribution(lengths):
    dens = sm.nonparametric.KDEUnivariate(lengths)
    dens.fit()
    return dens


def get_dnorm_val(dens, len_sent):
    # prob = dens.evaluate(len_sent)
    prob = math.sqrt(dens.evaluate(len_sent))
    return prob


# Single Token Probability: sTP
def single_token_probability(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_sTP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_sTP[i] = 1 - min([(max(m_pool[i][j].values())) for j in range(len_sent)])

    batch = np.argpartition(arr_sTP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Token Probability: tTP
def total_token_probability(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_tTP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_tTP[i] = sum([1 - (max(m_pool[i][j].values())) for j in range(len_sent)])

    batch = np.argpartition(arr_tTP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Normalized Token Probability: nTP
def normalized_token_probability(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_nTP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_nTP[i] = (
            sum([1 - (max(m_pool[i][j].values())) for j in range(len_sent)]) / len_sent
        )

    batch = np.argpartition(arr_nTP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Single Token Margin: sTM
def single_token_margin(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_sTM = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_sTM[i] = 1 - min(
            [
                max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2]
                for j in range(len_sent)
            ]
        )

    batch = np.argpartition(arr_sTM * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Token Margin: tTM
def total_token_margin(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_tTM = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_tTM[i] = sum(
            [
                1 - (max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2])
                for j in range(len_sent)
            ]
        )
    batch = np.argpartition(arr_tTM * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [s for s in idx_pool if s not in idx_q]


# Normalized Token Margin: nTM
def normalized_token_margin(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_nTM = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_nTM[i] = (
            sum(
                [
                    1 - (max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2])
                    for j in range(len_sent)
                ]
            )
            / len_sent
        )

    batch = np.argpartition(arr_nTM * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [s for s in idx_pool if s not in idx_q]


# Single Token Entropy: nTE
def single_token_entropy(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_NTE = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_NTE[i] = max(
            [
                (-1) * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                for j in range(len_sent)
            ]
        )

    batch = np.argpartition(arr_NTE * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [s for s in idx_pool if s not in idx_q]


# Total Token Entropy: tTE
def total_token_entropy(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_tTE = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_tTE[i] = sum(
            [
                (-1) * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                for j in range(len_sent)
            ]
        )

    batch = np.argpartition(arr_tTE * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Normalized Token Entropy: nTE
def normalized_token_entropy(m_pool, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_nTE = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_nTE[i] = (
            sum(
                [
                    (-1)
                    * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                    for j in range(len_sent)
                ]
            )
            / len_sent
        )

    batch = np.argpartition(arr_nTE * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Single Assignment Probability: sAP
def single_assignment_probability(m_pool, y_pred, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_sAP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_sAP[i] = 1 - min([(m_pool[i][j][y_pred[i][j]]) for j in range(len_sent)])

    batch = np.argpartition(arr_sAP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Assignment Probability: tAP
def total_assignemnt_probability(m_pool, y_pred, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_tAP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_tAP[i] = sum([1 - m_pool[i][j][y_pred[i][j]] for j in range(len_sent)])

    batch = np.argpartition(arr_tAP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Normalized Assignment Probability: nAP
def normalized_assignment_probability(m_pool, y_pred, idx_pool, batch_size):
    num_sent = len(m_pool)
    arr_nAP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(m_pool[i])
        arr_nAP[i] = (
            sum([1 - m_pool[i][j][y_pred[i][j]] for j in range(len_sent)]) / len_sent
        )

    batch = np.argpartition(arr_nAP * (-1), batch_size - 1)[:batch_size].tolist()
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


def get_outlier_mask(cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool):
    method_name = cfg["outlier_method"].get("method_name", None)
    mask_out_token = np.zeros(len(embeddings_pool))

    def get_mask(scores, mask_quantile):
        threshold = np.quantile(scores, mask_quantile)
        outliers = np.where(scores > threshold)[0]
        mask_out_token = np.zeros(len(scores))
        mask_out_token[outliers] = 1
        return mask_out_token

    if method_name == "GLOSH":
        mask_out_token = get_mask(
            clusterer.outlier_scores_[-len(embeddings_pool) :],
            cfg["outlier_method"]["mask_quantile"],
        )
    elif method_name == "LOF":
        clf = LocalOutlierFactor(  # contamination='auto' sets threshold to 0.1
            **cfg["outlier_method"]["parameters"]
        ).fit(np.concatenate(embeddings_pool))
        mask_out_token = get_mask(
            clf.negative_outlier_factor_,
            cfg["outlier_method"]["mask_quantile"],
        )
    elif method_name is None:
        pass
    else:
        raise ValueError(
            "An outlier method from the following list must be specified: ['GLOSH','LOF',None]"
        )

    mask_out_sentence = [
        mask_out_token[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]
    return mask_out_sentence


# Density Normalized Positive Token Probability: dpTP
def dnorm_positive_token_probability(
    cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_dpTP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_dpTP[i] = (
            sum(
                [
                    1 - max(m_pool[i][j].values())
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
                ]
            )
        ) * get_dnorm_val(PDF, len_sent)

    batch = np.argpartition(arr_dpTP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Positive Token Probability: tpTP
def total_positive_token_probability(
    cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_tpTP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_tpTP[i] = sum(
            [
                1 - max(m_pool[i][j].values())
                for j in range(len_sent)
                if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
            ]
        )

    batch = np.argpartition(arr_tpTP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Positive Token Margin: tpTM
def total_positive_token_margin(
    cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_tpTM = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_tpTM[i] = sum(
            [
                1 - max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2]
                for j in range(len_sent)
                if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
            ]
        )

    batch = np.argpartition(arr_tpTM * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Densitiy Normalized Positive Token Margin: dpTM
def dnorm_positive_token_margin(
    cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_dpTM = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_dpTM[i] = (
            sum(
                [
                    1 - max(m_pool[i][j].values()) - sorted(m_pool[i][j].values())[-2]
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
                ]
            )
        ) * get_dnorm_val(PDF, len_sent)

    batch = np.argpartition(arr_dpTM * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Densitiy Normalized Positive Token Entropy: dpTE
def dnorm_positive_token_entropy(
    cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_dpTE = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_dpTE[i] = (
            sum(
                [
                    (-1)
                    * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
                ]
            )
        ) * get_dnorm_val(PDF, len_sent)

    batch = np.argpartition(arr_dpTE * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Positive Token Entropy: tpTE
def total_positive_token_entropy(
    cfg, embeddings_ann, embeddings_pool, y_ann, m_pool, idx_pool, batch_size
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_tpTE = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_tpTE[i] = sum(
            [
                (-1) * sum([p * math.log2(p) for p in m_pool[i][j].values() if p > 0])
                for j in range(len_sent)
                if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
            ]
        )

    batch = np.argpartition(arr_tpTE * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Density Normalized Positive Assignment Probability: dpAP
def dnorm_positive_assignment_probability(
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_dpAP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_dpAP[i] = (
            sum(
                [
                    1 - m_pool[i][j][y_pred[i][j]]
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
                ]
            )
        ) * get_dnorm_val(PDF, len_sent)

    batch = np.argpartition(arr_dpAP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Total Positive Assignment Probability: tpAP
def total_positive_assignment_probability(
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_tpAP = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_tpAP[i] = sum(
            [
                1 - m_pool[i][j][y_pred[i][j]]
                for j in range(len_sent)
                if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
            ]
        )

    batch = np.argpartition(arr_tpAP * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Positive Annotation Selection: PAS
def positive_annotation_selection(
    cfg, embeddings_ann, embeddings_pool, y_ann, idx_pool, batch_size
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

    if cfg.get("plot", False):
        plot_umap_hdbscan(clusters_pool, embeddings_pool, experiment_dir, y_ann)

    mask_out_sentence = get_outlier_mask(
        cfg, clusterer, embeddings_ann, embeddings_pool, sent_idx_pool
    )

    n_ent = max(count_clusters.items(), key=operator.itemgetter(1))[0]
    num_sent = len(embeddings_pool)
    arr_PAS = np.zeros(num_sent)

    for i in range(num_sent):
        len_sent = len(embeddings_pool[i])
        arr_PAS[i] = (
            sum(
                [
                    1
                    for j in range(len_sent)
                    if (clusters_pool[i][j] != n_ent) or (mask_out_sentence[i][j] == 1)
                ]
            )
        ) * get_dnorm_val(PDF, len_sent)

    batch = np.argpartition(arr_PAS * (-1), batch_size - 1)[:batch_size].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]


# Random Selection: RS
def random_selection(idx_pool, batch_size, seed):
    random.seed(a=seed, version=2)

    idx_q = random.sample(idx_pool, batch_size)
    return idx_q, [s for s in idx_pool if s not in idx_q]


# Longest Sentence Selection: LSS
def longest_sentence_selection(sent_lenghts, idx_pool, batch_size):
    batch = np.argpartition(np.array(sent_lenghts) * (-1), batch_size - 1)[
        :batch_size
    ].tolist()
    idx_q = [idx_pool[i] for i in batch]

    return idx_q, [i for i in idx_pool if i not in idx_q]

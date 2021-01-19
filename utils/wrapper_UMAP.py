import numpy as np
import random

import umap
import hdbscan

from collections import Counter
from itertools import accumulate

LOW_MEM = False


def umap_r_embeddings(
    embeddings_ann, embeddings_pool, n_comp=200, neig=20, min_dist=0.1, seed=29
):

    umap_embedding = umap.UMAP(
        n_neighbors=neig,
        min_dist=min_dist,
        n_components=n_comp,
        random_state=seed,
        low_memory=LOW_MEM,
    )

    embeddings_ann_flat = [word.numpy() for sent in embeddings_ann for word in sent]
    embeddings_pool_flat = [word.numpy() for sent in embeddings_pool for word in sent]

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_len_ann = [0] + [len(sent) for sent in embeddings_ann]

    sent_idx_pool = list(accumulate(sent_len_pool))
    sent_idx_ann = list(accumulate(sent_len_ann))

    X_ = embeddings_ann_flat + embeddings_pool_flat

    embeddings = umap_embedding.fit_transform(X_)

    c_embeddings_ann = embeddings[: len(embeddings_ann_flat)]
    c_embeddings_pool = embeddings[len(embeddings_ann_flat) :]

    ur_embeddings_ann = [
        c_embeddings_ann[sent_idx_ann[i - 1] : sent_idx_ann[i]]
        for i in range(1, len(sent_idx_ann))
    ]
    ur_embeddings_pool = [
        c_embeddings_pool[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    return ur_embeddings_ann, ur_embeddings_pool


def ss_umap_r_embeddings(
    embeddings_ann,
    embeddings_pool,
    y_ann,
    tag_dict,
    n_comp=200,
    neig=20,
    min_dist=0.1,
    seed=29,
):
    umap_embedding = umap.UMAP(
        n_components=n_comp,
        n_neighbors=neig,
        min_dist=min_dist,
        random_state=seed,
        low_memory=LOW_MEM,
    )

    embeddings_ann_flat = [word.numpy() for sent in embeddings_ann for word in sent]
    embeddings_pool_flat = [word.numpy() for sent in embeddings_pool for word in sent]

    tags_ = [tag_dict[tag] for sent in y_ann for tag in sent]

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_len_ann = [0] + [len(sent) for sent in embeddings_ann]

    sent_idx_pool = list(accumulate(sent_len_pool))
    sent_idx_ann = list(accumulate(sent_len_ann))

    X_ = embeddings_ann_flat + embeddings_pool_flat
    y_ = tags_ + [-1 for word in embeddings_pool_flat]

    embeddings = umap_embedding.fit_transform(X_, y=y_)

    c_embeddings_ann = embeddings[: len(embeddings_ann_flat)]
    c_embeddings_pool = embeddings[len(embeddings_ann_flat) :]

    ss_umap_r_embeddings_ann = [
        c_embeddings_ann[sent_idx_ann[i - 1] : sent_idx_ann[i]]
        for i in range(1, len(sent_idx_ann))
    ]
    ss_umap_r_embeddings_pool = [
        c_embeddings_pool[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    return ss_umap_r_embeddings_ann, ss_umap_r_embeddings_pool


def ss_umap_r_hdbscan_c(
    embeddings_ann,
    embeddings_pool,
    y_ann,
    tag_dict,
    seed=29,
    n_comp=100,
    neig=40,
    min_dist=0.0,
    min_c_size=1000,
    min_samp=200,
    c_eps=0.2,
    **kwargs
):

    umap_embedding = umap.UMAP(
        n_components=n_comp,
        n_neighbors=neig,
        min_dist=min_dist,
        random_state=seed,
        low_memory=LOW_MEM,
    )

    embeddings_ann_flat = [np.array(word) for sent in embeddings_ann for word in sent]
    embeddings_pool_flat = [np.array(word) for sent in embeddings_pool for word in sent]

    tags_ = [tag_dict[tag] for sent in y_ann for tag in sent]

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_len_ann = [0] + [len(sent) for sent in embeddings_ann]

    sent_idx_pool = list(accumulate(sent_len_pool))
    sent_idx_ann = list(accumulate(sent_len_ann))

    X_ = embeddings_ann_flat + embeddings_pool_flat
    y_ = tags_ + [-1 for word in embeddings_pool_flat]

    embeddings = umap_embedding.fit_transform(X_, y=y_)

    c_embeddings_ann = embeddings[: len(embeddings_ann_flat)]
    c_embeddings_pool = embeddings[len(embeddings_ann_flat) :]

    ss_umap_r_embeddings_ann = [
        c_embeddings_ann[sent_idx_ann[i - 1] : sent_idx_ann[i]]
        for i in range(1, len(sent_idx_ann))
    ]
    ss_umap_r_embeddings_pool = [
        c_embeddings_pool[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_c_size,
        min_samples=min_samp,
        cluster_selection_epsilon=c_eps,
    )
    clusters = clusterer.fit_predict(embeddings)

    c_embeddings_ann = clusters[: len(embeddings_ann_flat)]
    c_embeddings_pool = clusters[len(embeddings_ann_flat) :]

    count_clusters = dict(Counter(c_embeddings_pool))

    clusters_ann = [
        c_embeddings_ann[sent_idx_ann[i - 1] : sent_idx_ann[i]]
        for i in range(1, len(sent_idx_ann))
    ]

    clusters_pool = [
        c_embeddings_pool[sent_idx_pool[i - 1] : sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]

    return (
        ss_umap_r_embeddings_ann,
        ss_umap_r_embeddings_pool,
        clusters_ann,
        clusters_pool,
        clusterer,
        count_clusters,
    )

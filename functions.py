import copy
import numpy as np
from itertools import accumulate

import utils.wrapper_UMAP as umap_
from utils.functions_AL import *
import math


def get_batch_size(cfg, iteration, pool_sent, total_sent):
    increment = cfg["increment_cons"]
    init_size = cfg["initial_size"]

    if isinstance(init_size, str):
        init_size = math.ceil(float(init_size[1:]) * (total_sent / 100))
    else:
        init_size = init_size

    if isinstance(increment, str):
        if increment[:3] == "exp":
            batch_size = int(math.ceil(float(increment[3:]) * (2**iteration)))
        elif increment[0] == "p":
            batch_size = int(
                iteration *
                math.ceil(float(increment[1:]) * (total_sent / 100)))
        elif increment[:2] == "cp":
            batch_size = int(
                math.ceil(float(increment[2:]) * (total_sent / 100)))
        else:
            raise ValueError("Unkown type incremet!")

    else:
        raise ValueError("Unkown type {increment size}/{batch size}!")

    if batch_size > pool_sent:
        batch_size = pool_sent

    return batch_size


def stopping_criteria(cfg, iteration, pool_sent, total_sent, f1):
    # batch_size = get_batch_size(cfg, iteration, pool_sent, total_sent)
    print("pool_sent", pool_sent)
    sc = cfg["stopping_criteria"]
    if sc == "":
        if pool_sent == 0:
            return True
        else:
            return False
    elif sc[:2] == "ge":
        if pool_sent <= math.ceil((total_sent * sc[2:]) / 100):
            return True
        else:
            return False
    else:
        raise ValueError("Unkown type stopping_criteria!")


def pca_r_embeddings(embeddings_ann, embeddings_pool, n_comp=200, seed=29):
    embeddings_ann_flat = [
        word.numpy() for sent in embeddings_ann for word in sent
    ]
    embeddings_pool_flat = [
        word.numpy() for sent in embeddings_pool for word in sent
    ]

    sent_len_pool = [0] + [len(sent) for sent in embeddings_pool]
    sent_len_ann = [0] + [len(sent) for sent in embeddings_ann]

    sent_idx_pool = list(accumulate(sent_len_pool))
    sent_idx_ann = list(accumulate(sent_len_ann))

    X_ = np.array(embeddings_ann_flat + embeddings_pool_flat)
    if np.isinf(X_).any():
        print("inf: ", X_[np.isinf(X_) == True])

    if np.isnan(X_).any():
        print("nan: ", X_[np.isnan(X_) == True])

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_comp, random_state=seed)
    embeddings = pca.fit_transform(X_)

    c_embeddings_ann = embeddings[:len(embeddings_ann_flat)]
    c_embeddings_pool = embeddings[len(embeddings_ann_flat):]

    pca_r_embeddings_ann = [
        c_embeddings_ann[sent_idx_ann[i - 1]:sent_idx_ann[i]]
        for i in range(1, len(sent_idx_ann))
    ]
    pca_r_embeddings_pool = [
        c_embeddings_pool[sent_idx_pool[i - 1]:sent_idx_pool[i]]
        for i in range(1, len(sent_idx_pool))
    ]
    print("Variance Explained:",
          list(accumulate(pca.explained_variance_ratio_)))

    return pca_r_embeddings_ann, pca_r_embeddings_pool


def reduce_embeddings(cfg, embeddings_train, embeddings_test):
    cfg_init_reduction = cfg["init_reduction"]
    if cfg_init_reduction["type"] == "pca":
        cfg_init_pca = cfg_init_reduction["pca"]
        embeddings_train_pca, embeddings_test_pca = pca_r_embeddings(
            embeddings_train,
            embeddings_test,
            n_comp=cfg_init_pca["dimension"],
            seed=cfg["seed"],
        )
        cfg.update({"ft_vec_dim": embeddings_train_pca[0][0].shape[-1]})
        return embeddings_train_pca, embeddings_test_pca

    elif cfg_init_reduction["type"] == "umap":
        cfg_init_umap = cfg_init_reduction["umap"]
        embeddings_train_ur, embeddings_test_ur = umap_.umap_r_embeddings(
            embeddings_train,
            embeddings_test,
            n_comp=cfg_init_umap["dimension"],
            neig=cfg_init_umap["neig"],
            min_dist=cfg_init_umap["min_dist"],
            seed=cfg["seed"],
        )
        cfg.update({"ft_vec_dim": embeddings_train_ur[0][0].shape[-1]})
        return embeddings_train_ur, embeddings_test_ur

    else:
        print("given reduction type is not defined --> no reduction")
        cfg.update({"ft_vec_dim": embeddings_train[0][0].shape[-1]})
        return embeddings_train, embeddings_test


def query(
    cfg,
    crf_trained,
    iteration,
    idx_pool,
    idx_ann,
    Xi_pool,
    embeddings_train,
    y_train,
    Xi_pool_2nd=[[]],
):
    if len(idx_pool) < 2:
        print("Batch size: ", len(idx_pool), "at iteration ", iteration)
        return idx_pool, []

    else:
        f_dict = cfg["method_dict"]
        active_learner = cfg["method"]
        batch_size = get_batch_size(cfg, iteration, len(idx_pool),
                                    len(y_train))
        print("Batch size: ", batch_size, "at iteration ", iteration)
        if active_learner == "rs":
            idx_q, idx_pool = f_dict[active_learner](idx_pool, batch_size,
                                                     cfg["seed"])

        elif active_learner in [
                "te",
                "tp",
                "tm",
                "tte",
                "ttp",
                "ttm",
                "nte",
                "ntp",
                "ntm",
        ]:
            mi_pool = crf_trained.predict_marginals(Xi_pool)
            idx_q, idx_pool = f_dict[active_learner](mi_pool, idx_pool,
                                                     batch_size)

        elif active_learner in ["ap", "tap", "nap"]:
            if cfg["generator"]:
                Xi_pool_ = Xi_pool_2nd
            else:
                Xi_pool_ = copy.deepcopy(Xi_pool)

            mi_pool = crf_trained.predict_marginals(Xi_pool)
            yi_pool = crf_trained.predict(Xi_pool_)
            idx_q, idx_pool = f_dict[active_learner](mi_pool, yi_pool,
                                                     idx_pool, batch_size)

        elif active_learner in ["pte", "ptp", "ptm", "pap"]:
            embeddings_ann = [embeddings_train[x] for x in idx_ann]
            embeddings_pool = [embeddings_train[x] for x in idx_pool]
            y_ann = [y_train[x] for x in idx_ann]
            if active_learner == "pap":
                if cfg["generator"]:
                    Xi_pool_ = Xi_pool_2nd
                else:
                    Xi_pool_ = copy.deepcopy(Xi_pool)

                mi_pool = crf_trained.predict_marginals(Xi_pool)
                yi_pool = crf_trained.predict(Xi_pool_)
                idx_q, idx_pool = f_dict[active_learner](
                    cfg,
                    embeddings_ann,
                    embeddings_pool,
                    y_ann,
                    yi_pool,
                    mi_pool,
                    idx_pool,
                    batch_size,
                )
            else:
                mi_pool = crf_trained.predict_marginals(Xi_pool)
                idx_q, idx_pool = f_dict[active_learner](
                    cfg,
                    embeddings_ann,
                    embeddings_pool,
                    y_ann,
                    mi_pool,
                    idx_pool,
                    batch_size,
                )

        elif active_learner == "mes":
            embeddings_ann = [embeddings_train[x] for x in idx_ann]
            embeddings_pool = [embeddings_train[x] for x in idx_pool]
            y_ann = [y_train[x] for x in idx_ann]

            idx_q, idx_pool = f_dict[active_learner](
                cfg,
                embeddings_ann,
                embeddings_pool,
                y_ann,
                idx_pool,
                batch_size,
            )
        elif active_learner == "lss":
            embeddings_pool = [embeddings_train[x] for x in idx_pool]
            idx_q, idx_pool = f_dict[active_learner](
                [len(sent) for sent in embeddings_pool],
                idx_pool,
                batch_size,
            )
        else:
            raise ValueError("given acitve learner is not defined")
        return idx_q, idx_pool

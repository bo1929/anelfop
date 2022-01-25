import os
import string
from tqdm import tqdm

import sklearn_crfsuite


def feature_selector(
    feature_cfg,
    word,
    i,
    j,
    conf_switch,
    pos=[[]],
    embeddings=[[]],
):
    feature_dict_embed = {}
    if embeddings:
        for k in range(0, len(embeddings[i][j])):
            feature_dict_embed[conf_switch + ":vec" + str(k + 1)] = embeddings[i][j][k]

    feature_pos = {}
    if conf_switch + ":POS" in feature_cfg[conf_switch]:
        feature_pos = {conf_switch + ":POS": str(pos[i][j])}

    features_word = {
        conf_switch + ":wordLower": word.lower(),
        conf_switch + ":last2letter": word[-2:],
        conf_switch + ":last3letter": word[-3:],
        conf_switch + ":except2letter": word[:-2],
        conf_switch + ":except3letter": word[:-3],
        conf_switch + ":isTitle": word.istitle(),
        conf_switch + ":isLower": word.islower(),
        conf_switch + ":isDigit": word.isdigit(),
    }

    feature_dict = {}
    feature_dict.update(feature_dict_embed)
    feature_dict.update(features_word)
    feature_dict.update(feature_pos)

    return {
        i: feature_dict.get(i)
        for i in feature_cfg[conf_switch]
        if i in feature_dict.keys()
    }


# Do not pass [CLS] [SEP] tokens and their embeddings \
#  only next word and previous word are considered \
#  sent, word_embd parameters are common for all i.


def word2features(feature_cfg, sents, i, j, **kwargs):
    features = feature_selector(feature_cfg, sents[i][j], i, j, "0", **kwargs)
    features["0:bias"] = 1.0

    if j > 0:
        if j == 1:
            features["-1:BOS"] = True

        features.update(
            feature_selector(feature_cfg, sents[i][j - 1], i, j - 1, "-1", **kwargs)
        )
    else:
        features["0:BOS"] = True

    if j < len(sents[i]) - 1:
        if j == len(sents[i]) - 2:
            features["+1:EOS"] = True

        features.update(
            feature_selector(feature_cfg, sents[i][j + 1], i, j + 1, "+1", **kwargs)
        )
    else:
        features["0:EOS"] = True
    return features


# Returns generator for each sentences, slower but memory efficient.
def sent2dict_list(feature_cfg, sents, i, **kwargs):
    return [
        word2features(feature_cfg, sents, i, j, **kwargs) for j in range(len(sents[i]))
    ]


def sent2features(feature_cfg, sents, generator=False, **kwargs):
    if generator:
        return (
            sent2dict_list(feature_cfg, sents, i, **kwargs)
            for i in tqdm(range(len(sents)))
        )
    else:
        return [
            sent2dict_list(feature_cfg, sents, i, **kwargs)
            for i in tqdm(range(len(sents)))
        ]


def tag2labels(tags, generator=False):
    if generator:
        return (tags[i] for i in tqdm(range(len(tags))))
    else:
        return [tags[i] for i in tqdm(range(len(tags)))]


def train_crf(cfg, X_train, y_train):
    cfg_crf = cfg["CRF"]
    crf = sklearn_crfsuite.CRF(
        algorithm=cfg_crf["algorithm"],
        c1=cfg_crf["c1"],
        c2=cfg_crf["c2"],
        max_iterations=cfg_crf["max_iterations"],
        all_possible_transitions=cfg_crf["allow_all_transitions"],
        all_possible_states=cfg_crf["allow_all_states"],
        verbose=True,
    )
    return crf.fit(X_train, y_train)

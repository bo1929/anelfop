import os
import logging
import pprint

from methodsAL import *

import json
import yaml
import joblib
import itertools
import pickle as pkl


def load_data(cfg):
    a = 1
    name = cfg["data_set"]["name"]

    tknzd_sent = []
    with open(cfg["data_directory"] + name + "_train.tokenized") as tokenized_file:
        tknzd_sent = json.load(tokenized_file)

    tags = []
    with open(cfg["data_directory"] + name + "_train.tags") as tags_file:
        tags = json.load(tags_file)

    tknzd_sent_test = []
    with open(cfg["data_directory"] + name + "_test.tokenized") as tokenized_file:
        tknzd_sent_test = json.load(tokenized_file)

    tags_test = []
    with open(cfg["data_directory"] + name + "_test.tags") as tags_file:
        tags_test = json.load(tags_file)

    if cfg["data_set"]["pos"]:
        pos_tags = []
        with open(cfg["data_directory"] + name + "_train.pos") as pos_file:
            pos_tags = json.load(pos_file)

        pos_tags_test = []
        with open(cfg["data_directory"] + name + "_test.pos") as pos_file:
            pos_tags_test = json.load(pos_file)
    else:
        pos_tags, pos_tags_test = tags, tags_test

    named_entities = list(set(list(itertools.chain.from_iterable(tags))))
    tag_dict = {named_entities[i]: i for i in range(len(named_entities))}
    cfg.update({"tag_dict": tag_dict})

    return_train = [tknzd_sent, tags, pos_tags]
    return_test = [tknzd_sent_test, tags_test, pos_tags_test]

    return return_train, return_test


def cfg_from_file(filename):
    """
    Load a config from file filename
    """
    with open(filename, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return yaml_cfg


def load_config_from(filename="./config.yaml", AL=True):
    cfg = cfg_from_file(filename)

    expt_dir = cfg["main_directory"] + "expt_results"
    if not os.path.exists(expt_dir):
        os.mkdir(expt_dir)
    
    if AL == False:
        cfg["method"] = "passive"
        results_dir = os.path.join(expt_dir, "results_passive", "")
    else:
        results_dir = os.path.join(
             expt_dir, "results_active" + "_" + str(cfg["seed"]), ""
        )

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    cfg.update({"results_directory": results_dir})
    cfg_init_reduction = cfg["init_reduction"]
    init_reduction_type = cfg_init_reduction["type"]

    if (
        init_reduction_type not in cfg_init_reduction.keys()
        or init_reduction_type == "type"
    ):
        init_reduction_pfix = ""
    else:
        init_reduction_pfix = init_reduction_type + str(
            cfg_init_reduction[init_reduction_type]["dimension"]
        )

    experiment_dir = os.path.join(
        results_dir,
        cfg["method"]
        + "_"
        + cfg["data_set"]["name"]
        + "_"
        + os.path.split(cfg["pretrained_model"])[-1].replace(".", "")
        + "_"
        + cfg["embedding_type"]
        + "_"
        + init_reduction_pfix,
        "",
    )
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    cfg.update({"experiment_directory": experiment_dir})

    if not os.path.exists(cfg["data_directory"]):
        raise ValueError("could not find data_directory given in the config file")

    method_dict = {
        "rs": rs,
        "te": te,
        "tp": tp,
        "tm": tm,
        "tte": tte,
        "ttp": ttp,
        "ttm": ttm,
        "nte": nte,
        "ntp": ntp,
        "ntm": ntm,
        "ap": ap,
        "tap": tap,
        "nap": nap,
        "pte": pte,
        "ptp": ptp,
        "ptm": ptm,
        "pap": pap,
        "mes": mes,
        "lss": lss,
    }
    cfg.update({"method_dict": method_dict})

    print(cfg["data_set"]["name"])
    return cfg


################


def load_ft_config(cfg):
    experiment_dir = cfg["experiment_directory"]
    with open(os.path.join(experiment_dir, "features_config.yaml"), "r") as f:
        result = yaml.load(f, Loader=yaml.FullLoader)
    return result


################


def save_crf_model(cfg, model, iteration):
    experiment_dir = cfg["experiment_directory"]
    if iteration != 0:
        models_dir = os.path.join(experiment_dir, "updated_models", "")
        name = "models_iter" + str(iteration)
    else:
        models_dir = os.path.join(experiment_dir, "passive_model", "")
        name = "passive"

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    with open(os.path.join(models_dir, name), "wb") as outfile:
        joblib.dump(value=model, filename=outfile)


################


def load_crf_model(cfg, iteration):
    experiment_dir = cfg["experiment_directory"]
    models_dir = os.path.join(experiment_dir, "updated_models", "")

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    with open(
        os.path.join(models_dir, "models_iter" + str(iteration)), "rb"
    ) as outfile:
        model = joblib.load(filename=outfile)
    return model


################


def write_ft_config(cfg):
    experiment_dir = cfg["experiment_directory"]
    dim = cfg["ft_vec_dim"]
    have_pos = cfg["data_set"]["pos"]

    print("embeddings vector dimension:", dim)

    span = ["-1", "0", "+1"]

    features_dict = {}
    features_all = []
    if have_pos:
        features_all += ["POS"]
    if dim != 0:
        features_all += ["vec" + str(i) for i in range(1, dim + 1)]

    features_all += [
        "wordLower",
        "last2letter",
        "last3letter",
        "isTitle",
        "isLower",
        "isDigit",
    ]
    features_all += [
        "BOS",
        "EOS",
        "bias",
    ]

    for i in span:
        features_dict[i] = [i + ":" + f for f in features_all]

    with open(experiment_dir + "features_config.yaml", "w") as outfile:
        yaml.dump(features_dict, outfile, default_flow_style=False)


def save_results(cfg, stats_query, f1_scores, query_indexes, query_sent_len):
    experiment_dir = cfg["experiment_directory"]

    with open(os.path.join(experiment_dir, "stats_query"), "wb") as outfile:
        pkl.dump(stats_query, outfile)

    with open(os.path.join(experiment_dir, "f1_scores"), "wb") as outfile:
        pkl.dump(f1_scores, outfile)

    with open(os.path.join(experiment_dir, "query_indexes"), "wb") as outfile:
        pkl.dump(query_indexes, outfile)

    with open(os.path.join(experiment_dir, "query_sent_len"), "wb") as outfile:
        pkl.dump(query_sent_len, outfile)

    with open(experiment_dir + "cfg_experiment.yaml", "w") as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

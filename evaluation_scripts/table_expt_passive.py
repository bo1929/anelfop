import itertools
import pickle

import glob
import os

import numpy as np
from tabulate import tabulate

if not os.path.exists("../evaluations/"):
    os.mkdir("../evaluations/")

if not os.path.exists("../evaluations/passive_tables/"):
    os.mkdir("../evaluations/passive_tables/")

# Read results and group them.

f1_results_paths = glob.glob(
    "../expt_results/results_passive/**/f1_scores", recursive=True
)

details_tuple = []
for f1path in f1_results_paths:
    model_name = (
        os.path.normpath(f1path)
        .split(os.sep)[-2]
        .replace("NCBI_disease", "NCBI-disease")
        .replace("Bio_ClinicalBERT", "Bio-ClinicalBERT")
    )

    temp_details = model_name.split("_")
    corpus, pre_model, embedding_type, embedding_dimension = (
        temp_details[1],
        temp_details[2],
        temp_details[3],
        temp_details[4],
    )
    details_tuple.append(
        [f1path, corpus, pre_model, embedding_type, embedding_dimension]
    )


def key_func(i):
    return lambda x: x[i]


details_tuple.sort(key=key_func(1))
for key, group1 in itertools.groupby(details_tuple, key_func(1)):
    table = []
    group1 = list(group1)
    group1.sort(key=key_func(2))
    for key2, group2 in itertools.groupby(group1, key_func(2)):
        group2 = list(group2)
        group2.sort(key=key_func(3))
        for key3, group3 in itertools.groupby(group2, key_func(3)):
            group3 = list(group3)
            group3.sort(key=key_func(4))
            for item in group3:
                if item[4] == "":
                    item[4] = "768"
                else:
                    item[4] = item[4][:3] + "-" + item[4][3:]
                with open(item[0], "rb") as openfile:
                    table.append(
                        [key2, key3, item[4]] + list(np.round(pickle.load(openfile), 3))
                    )
    header_ = ["pre-trained model", "embedding type", "embedding dimension", "f1-score"]
    with open(
        "../evaluations/passive_tables/" + key + "_table_passive_model.tex", "w"
    ) as file1:
        file1.write(
            tabulate(
                table,
                headers=header_,
                tablefmt="latex",
            )
        )
    with open(
        "../evaluations/passive_tables/" + key + "_table_passive_model.md", "w"
    ) as file2:
        file2.write(
            tabulate(
                table,
                headers=header_,
                tablefmt="github",
            )
        )

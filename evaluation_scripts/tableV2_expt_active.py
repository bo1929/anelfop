import itertools
import pickle

import glob
import os

import numpy as np

from tabulate import tabulate

if not os.path.exists("../evaluations/"):
    os.mkdir("../evaluations/")

if not os.path.exists("../evaluations/active_tables/"):
    os.mkdir("../evaluations/active_tables/")

genus_key_dict = {
    "tp": 0,
    "ttp": 1,
    "ntp": 2,
    "ptp": 3,
    "tm": 4,
    "ttm": 5,
    "ntm": 6,
    "ptm": 7,
    "te": 8,
    "tte": 9,
    "nte": 10,
    "pte": 11,
    "ap": 12,
    "tap": 13,
    "nap": 14,
    "pap": 15,
    "lss": 16,
    "rs": 17,
}

f1_results_paths = glob.glob(
    "../expt_results/results_active*/**/f1_scores", recursive=True
)

# token_results_paths = glob.glob(
#     "../expt_results/results_active*/**/query_sent_len", recursive=True
# )
details_tuple = []
for f1path in f1_results_paths:
    model_name = (
        os.path.normpath(f1path)
        .split(os.sep)[-2]
        .replace("NCBI_disease", "NCBI-disease")
        .replace("Bio_ClinicalBERT", "Bio-ClinicalBERT")
    )

    temp_details = model_name.split("_")
    method, corpus, pre_model, embedding_type, embedding_dimension = (
        temp_details[0],
        temp_details[1],
        temp_details[2],
        temp_details[3],
        temp_details[4],
    )
    details_tuple.append(
        [f1path, method, corpus, pre_model, embedding_type, embedding_dimension]
    )


def key_func(i):
    return lambda x: x[i]


genus_key = lambda row: genus_key_dict[row[0]]

details_tuple.sort(key=key_func(2))
for key, group1 in itertools.groupby(details_tuple, key_func(2)):
    table = []
    group1 = list(group1)
    group1.sort(key=key_func(1))
    for key2, group2 in itertools.groupby(group1, key_func(1)):
        temp = []
        temp1 = []
        for item in group2:
            if item[5] == "":
                item[5] = "768"
            else:
                item[5] = item[5][:3] + "-" + item[5][3:]
            with open(item[0], "rb") as openfile:
                temp.append(pickle.load(openfile))
            head, _ = os.path.split(item[0])
            with open(os.path.join(head, "query_sent_len"), "rb") as openfile:
                temp1.append([sum(query) for query in pickle.load(openfile)])
        tokenAvg = list(np.cumsum(np.mean(np.array(temp1), axis=0)).astype(int))
        f1Avg = list(np.mean(np.array(temp), axis=0))
        table.append(
            [item[1], item[3], item[4], item[5]]
            + list(map(str, list(zip(f1Avg, tokenAvg))))
        )

    table.sort(key=genus_key)
    header_ = [
        "AL method",
        "pre-trained model",
        "embedding type",
        "embedding dimension",
    ] + ["f1-score & sentence length " + str(i) for i in range(len(table[0]) - 4)]
    with open(
        "../evaluations/active_tables/" + key + "_tableV2_active_expt.tex", "w"
    ) as file1:
        file1.write(
            tabulate(
                table,
                headers=header_,
                tablefmt="latex",
            )
        )
    with open(
        "../evaluations/active_tables/" + key + "_tableV2_active_expt.md", "w"
    ) as file2:
        file2.write(
            tabulate(
                table,
                headers=header_,
                tablefmt="github",
            )
        )

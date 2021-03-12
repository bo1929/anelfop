import json
import os
import glob

import itertools
from collections import Counter
import numpy as np

from tabulate import tabulate

if not os.path.exists("../evaluations/"):
    os.mkdir("../evaluations/")

if not os.path.exists("../evaluations/corpus_stats/"):
    os.mkdir("../evaluations/corpus_stats/")


# tags_corpus_paths = glob.glob(
#     "../datasets/tokenized/*.tags", recursive=True
# )

tokenized_corpus_paths = glob.glob("../datasets/tokenized/*.tokenized", recursive=True)

###################

details_tuple = []
for corpusTokenizedPath in tokenized_corpus_paths:
    head, tail = os.path.split(corpusTokenizedPath)
    corpus_name, type = (
        tail.split(".")[0].replace("NCBI_disease", "NCBI-disease").split("_")
    )

    corpusTagPath = ".." + corpusTokenizedPath.split(".")[-2] + ".tags"
    details_tuple.append([corpus_name, type, corpusTokenizedPath, corpusTagPath])


def key_func(i):
    return lambda x: x[i]


details_tuple.sort(key=key_func(0))
for key, group1 in itertools.groupby(details_tuple, key_func(0)):

    def append_table(key, type, tags, table):
        tags_flat = [t for sent in tags for t in sent]
        total_token = len(tags_flat)
        total_sent = len(tags)

        count_tags_flat = dict(Counter(tags_flat))
        b_count = {
            tag[1:]: 0 
            for tag in count_tags_flat.keys()
            if tag[0] in ["B", "I"]
        }
        i_count ={
            tag[1:]: 0 
            for tag in count_tags_flat.keys()
            if tag[0] in ["B", "I"]
        } 
        b_count.update({
            tag[1:]: count_tags_flat[tag]
            for tag in count_tags_flat.keys()
            if tag[0] == "B"
        })
        i_count.update({
            tag[1:]: count_tags_flat[tag]
            for tag in count_tags_flat.keys()
            if tag[0] == "I"
        })
        count_tags_flat = {tag: b_count[tag] + i_count[tag] for tag in i_count.keys()}
        print(count_tags_flat)
        tags_PN = [[0 if tag == "O" else 1 for tag in sent] for sent in tags]
        avg_SL = np.mean([len(seq) for seq in tags_PN])
        avg_PToken = np.mean([sum(seq) for seq in tags_PN])
        sum_PToken = sum([sum(seq) for seq in tags_PN])
        AC_PToken = sum([1 if sum(seq) >= 1 else 0 for seq in tags_PN]) / len(tags_PN)
        DAC_PToken = sum([1 if sum(seq) >= 2 else 0 for seq in tags_PN]) / len(tags_PN)

        table.append(
            [
                key,
                type,
                total_sent,
                total_token,
                sum_PToken,
                sum_PToken / total_token,
                avg_SL,
                avg_PToken,
                AC_PToken,
                DAC_PToken,
            ]
            + [count / sum_PToken for count in list(count_tags_flat.values())]
        )

        return [
            "Corpus",
            "Partition",
            "TotalSentence",
            "TotalToken",
            "TotalPToken",
            "PercentPToken",
            "AvgSentenceLength",
            "AvgNumberPTokenPerSentence",
            "SentWithPToken",
            "SentWith2PToken",
        ] + ["Percent" + t_item[1:] for t_item in list(count_tags_flat.keys())]

    table = []
    all_tokenized = []
    all_tags = []
    for item in group1:
        print(item)
        with open(item[2], "rb") as openfile:
            tokenized_ = json.load(openfile)

        with open(item[3], "rb") as openfile:
            tags_ = json.load(openfile)

        all_tokenized = all_tokenized + tokenized_
        all_tags = all_tags + tags_

        append_table(key, item[1], tags_, table)

    header_ = append_table(key, "all", all_tags, table)

    table.sort(key=key_func(1))

    with open("../evaluations/corpus_stats/" + key + "_corpus_stats.tex", "w") as file1:
        file1.write(
            tabulate(
                table,
                headers=header_,
                tablefmt="latex",
            )
        )
    with open("../evaluations/corpus_stats/" + key + "_corpus_stats.md", "w") as file2:
        file2.write(
            tabulate(
                table,
                headers=header_,
                tablefmt="github",
            )
        )

import os
import pickle
import itertools

import numpy as np

import seaborn as sns
import matplotlib

from scipy.interpolate import interp1d

INTPLT = False

EXPERIMENTS = ["exp1_init16/experiment_no1", "exp1_init16/experiment_no2", "exp1_init16/experiment_no3"]
CORPUS = "s800"
AL__GENUS = "ap"

BATCH_CONST = 1


def get_parent_dir(x, depth=2):
    parent_dir = os.path.dirname(x)
    for i in range(1, depth):
        parent_dir = os.path.dirname(parent_dir)
    return parent_dir


def plot_single_genus(results_name, results_f1, cumnum_token, INTPLT=False):
    L = len(results_f1[0])
    batch_sizes = list(
        itertools.accumulate([16] + [BATCH_CONST * (2 ** i) for i in range(1, L)])
    )
    cumnum_sent = [batch_sizes for i in range(len(cumnum_token))]

    if INTPLT:
        results_f1_intplt = []
        cumnum_sent_intplt = []
        cumnum_token_intplt = []
        cumnum_token_common = []

        common_token_count = np.geomspace(
            cumnum_token[0][0],
            cumnum_token[0][-1],
            num=len(cumnum_token[0]),
        )
        for i in range(len(cumnum_token)):
            f_ctf1 = interp1d(
                cumnum_token[i],
                results_f1[i],
                kind="quadratic",
            )
            f_csct = interp1d(
                cumnum_sent[i],
                cumnum_token[i],
                kind="quadratic",
            )

            cumnum_token_common.append(common_token_count)
            results_f1_intplt.append(f_ctf1(common_token_count))
            cumnum_token_intplt.append(f_csct(cumnum_sent[i]))

        x_val = [cumnum_sent, cumnum_token_common, cumnum_sent]
        y_val = [results_f1, results_f1_intplt, cumnum_token_intplt]
    else:
        x_val = [cumnum_sent, cumnum_token, cumnum_sent]
        y_val = [results_f1, results_f1, cumnum_token]

    x_labels = [
        "number of annotated sentences",
        "number of annotated tokens",
        "number of annotated sentences",
    ]
    y_labels = ["f1-score", "f1-score", "number of annotated tokens"]

    x_scales = ["log", "log", "log"]
    y_scales = ["linear", "linear", "log"]

    basex = [2, 2, 2]
    basey = [10, 10, 2]

    marker_cyc = itertools.cycle(
        list(matplotlib.lines.Line2D.markers.keys())[: len(results_name)]
    )
    palette = itertools.cycle(sns.color_palette()[: len(results_name)])

    fig = matplotlib.figure.Figure(figsize=(18, 6), constrained_layout=False, dpi=700)
    ax_array = fig.subplots(1, 3, squeeze=False)

    for i in range(ax_array.shape[1]):
        for j in range(len(results_f1)):
            ax_array[0, i].plot(
                x_val[i][j],
                y_val[i][j],
                "o-",
                label=(results_name[j]),
                linewidth=0.7,
                ms=1,
                alpha=0.7,
                marker=next(marker_cyc),
                color=next(palette),
            )

        ax_array[0, i].set_xlabel(x_labels[i])
        ax_array[0, i].set_ylabel(y_labels[i])

        ax_array[0, i].set_xscale(x_scales[i], base=basex[i])
        ax_array[0, i].set_yscale(y_scales[i], base=basey[i])

    matplotlib.rcParams.update({"font.size": 12})

    sns.set_context("paper")
    sns.axes_style("ticks")

    handles, labels = ax_array[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", prop={"size": 18})
    fig.savefig("./here.png")


results_f1_all = []
cumnum_token_all = []

for experiment in EXPERIMENTS:
    result_path_all = os.path.join(
        get_parent_dir(os.path.realpath(__file__)),
        "results_active",
        experiment,
        CORPUS,
        "",
    )
    result_name_method = os.listdir(result_path_all)

    results_f1_ = []
    cumnum_token_ = []
    results_name = []

    for name_long in sorted(result_name_method):
        name = name_long.split("_")[0]
        if AL__GENUS in name:
            results_name.append(name)
            f1_score_path = os.path.join(result_path_all, name_long, "f1_scores")
            qsent_len_path = os.path.join(result_path_all, name_long, "query_sent_len")

            with (open(f1_score_path, "rb")) as openfile:
                results_f1_.append(np.array(pickle.load(openfile)).astype(float))

            with (open(qsent_len_path, "rb")) as openfile:
                qsent_len = pickle.load(openfile)
                cumnum_token_.append(
                    np.cumsum(
                        np.array([sum(query) for query in qsent_len]).astype(float)
                    )
                )
    results_f1_all.append(np.array(results_f1_))
    cumnum_token_all.append(np.array(cumnum_token_))

results_f1 = np.zeros(results_f1_all[0].shape)
cumnum_token = np.zeros(cumnum_token_all[0].shape)

for j in range(cumnum_token.shape[0]):
    for i in range(len(EXPERIMENTS)):
        cumnum_token[j] = cumnum_token[j] + cumnum_token_all[i][j]
        results_f1[j] = results_f1[j] + results_f1_all[i][j]
    cumnum_token[j] = cumnum_token[j] / len(EXPERIMENTS)
    results_f1[j] = results_f1[j] / len(EXPERIMENTS)

plot_single_genus(results_name, results_f1, cumnum_token, INTPLT=INTPLT)

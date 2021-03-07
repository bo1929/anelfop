import os
import sys
import pickle
import itertools

import numpy as np

import matplotlib as mpl
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.interpolate import pchip_interpolate, interp1d

print(sys.argv)

SMALL_SIZE = 15
MEDIUM_SIZE = 17
BIGGER_SIZE = 19

mpl.rc("font", size=SMALL_SIZE)  # controls default text sizes
mpl.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
mpl.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
mpl.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
mpl.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.7
mpl.rcParams["grid.color"] = "#cccccc"
mpl.rcParams["grid.linestyle"] = "-"

mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["xtick.major.width"] = 1.4
mpl.rcParams["xtick.minor.size"] = 2
mpl.rcParams["xtick.minor.width"] = 1

mpl.rcParams["ytick.major.size"] = 3
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["ytick.minor.size"] = 2
mpl.rcParams["ytick.minor.width"] = 1

if not os.path.exists("../evaluations/"):
    os.mkdir("../evaluations/")

if not os.path.exists("../evaluations/genus_plots/"):
    os.mkdir("../evaluations/genus_plots/")

OPT_DICT = {
    "sentence-f1": 0,
    "token-f1": 1,
    "sentence-token": 2,
    "query-gradient": 3,
}

EXPERIMENTS = [
    "results_active_112",
    "results_active_121",
    "results_active_211",
    "results_active_199",
    "results_active_227",
    "results_active_991",
    "results_active_272",
    "results_active_919",
    "results_active_722",
]

genus_dict = {
    "tp": ["tp", "ttp", "ntp", "ptp"],
    "tm": ["tm", "ttm", "ntm", "ptm"],
    "te": ["te", "tte", "nte", "pte"],
    "ap": ["ap", "tap", "nap", "pap"],
}

DIVIDE = False
BATCH_CONST = 1
INTPLT = False
PUT_LEGEND = False
ERROR_BAR = True

CORPUS = sys.argv[1]
GENUS = sys.argv[2]
OPTS = sys.argv[3:]

IDX = [OPT_DICT[opt] for opt in OPTS]

NAME = CORPUS + "_" + GENUS
for opt in OPTS:
    NAME += "_" + opt


def get_parent_dir(x, depth=2):
    parent_dir = os.path.dirname(x)
    for i in range(1, depth):
        parent_dir = os.path.dirname(parent_dir)
    return parent_dir


def plot_single_genus(
    name_output,
    results_name,
    batch_sizes,
    results_f1,
    cumnum_token,
    std_error,
    INTPLT=False,
):
    cumnum_sent = [batch_sizes for i in range(len(cumnum_token))]
    queries = [
        [j for j in range(1, len(cumnum_token[0]))] for i in range(len(cumnum_token))
    ]
    common_f1 = np.arange(min(results_f1[0]), max(results_f1[0]), 0.1)
    common_token_count = np.geomspace(
        cumnum_token[0][0],
        cumnum_token[0][-1],
        num=len(cumnum_token[0]),
    )

    if INTPLT:
        results_f1_intplt = np.ndarray(shape=(len(cumnum_token), len(cumnum_token[0])))
        cumnum_token_intplt = []
        cumnum_token_common = []

        for i in range(len(cumnum_token)):
            """
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
            intp_ctf1 = f_ctf1(common_token_count)
            intp_fsct = f_csct(cumnum_sent[i])
            """
            intp_ctf1 = pchip_interpolate(
                cumnum_token[i], results_f1[i], common_token_count
            )
            intp_fsct = pchip_interpolate(
                cumnum_sent[i], cumnum_token[i], cumnum_sent[i]
            )

            cumnum_token_common.append(common_token_count)
            results_f1_intplt[i] = intp_ctf1
            cumnum_token_intplt.append(intp_fsct)

        gradient = (results_f1_intplt[:, 1:] - results_f1_intplt[:, :-1]) / (
            (common_token_count[1:] - common_token_count[:-1])
        )

        x_val = [cumnum_sent, cumnum_token_common, cumnum_sent, queries]
        y_val = [results_f1, results_f1_intplt, cumnum_token_intplt, gradient]
    else:
        gradient = (results_f1[:, 1:] - results_f1[:, :-1]) / (
            (cumnum_token[:, 1:] - cumnum_token[:, :-1])
        )

        x_val = [cumnum_sent, cumnum_token, cumnum_sent, queries]
        y_val = [results_f1, results_f1, cumnum_token, gradient]

    x_labels = [
        "number of annotated sentences",
        "number of annotated tokens",
        "number of annotated sentences",
        "number of queries",
    ]
    y_labels = [
        "f1-score",
        "f1-score",
        "number of annotated tokens",
        "f1-score gradient",
    ]

    x_scales = ["log", "log", "log", "linear"]
    y_scales = ["linear", "linear", "log", "linear"]

    basex = [2, 2, 2, 10]
    basey = [10, 10, 2, 10]

    show_error = [True, True, False, True]

    len_ = len(results_name)

    marker_cyc = itertools.cycle(list(mpl.lines.Line2D.filled_markers)[-len_:])
    palette = itertools.cycle(sns.color_palette("Set2")[:len_])

    fig = mpl.figure.Figure(
        figsize=(12 * len(OPTS), 9), constrained_layout=False, dpi=100
    )
    ax_array = fig.subplots(1, len(OPTS), squeeze=False)
    count = 0
    for i in IDX:
        for j in range(len(results_f1)):
            ctemp = next(palette)
            ax_array[0, count].plot(
                x_val[i][j],
                y_val[i][j],
                # "o-",
                label=(results_name[j]),
                linewidth=1.2,
                ms=7.0,
                alpha=1.0,
                marker=next(marker_cyc),
                color=ctemp,
            )
            if ERROR_BAR and show_error[i]:
                ax_array[0, count].errorbar(
                    x_val[i][j],
                    y_val[i][j],
                    std_error[j],
                    alpha=0.7,
                    color=ctemp,
                )

    ax_array[0, count].set_xlabel(x_labels[i])
    ax_array[0, count].set_ylabel(y_labels[i])

    ax_array[0, count].set_xscale(x_scales[i], base=basex[i])
    ax_array[0, count].set_yscale(y_scales[i], base=basey[i])

    # ax_array[0, count].spines['right'].set_visible(False)
    # ax_array[0, count].spines['top'].set_visible(False)

    # if x_scales[i] == "log":
    #         locmaj = mpl.ticker.LogLocator(base=basex[i], numticks=8)
    #         ax_array[0, count].xaxis.set_major_locator(locmaj)

    count += 1

    sns.set_context("paper")
    sns.axes_style("ticks")

    if PUT_LEGEND:
        handles, labels = ax_array[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", prop={"size": 18})

    fig.savefig(
        "../evaluations/genus_plots/" + name_output + ".svg", bbox_inches="tight", pad_inches=0.1
    )


results_f1_all = []
cumnum_token_all = []

for experiment in EXPERIMENTS:
    result_path_all = os.path.join(
        get_parent_dir(os.path.realpath(__file__)),
        "expt_results",
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
        if GENUS in name:
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

results_f1_all = np.array(results_f1_all)
cumnum_token_all = np.array(cumnum_token_all)

results_f1 = np.zeros(results_f1_all.shape[1:])
results_error = np.zeros(results_f1_all.shape[1:])
cumnum_token = np.zeros(cumnum_token_all.shape[1:])

for j in range(cumnum_token.shape[0]):
    for i in range(len(EXPERIMENTS)):
        cumnum_token[j] = cumnum_token[j] + cumnum_token_all[i][j]
        results_f1[j] = results_f1[j] + results_f1_all[i][j]
    results_error[j] = np.std(results_f1_all[:, j, :], axis=0)
    cumnum_token[j] = cumnum_token[j] / len(EXPERIMENTS)
    results_f1[j] = results_f1[j] / len(EXPERIMENTS)

pass_name = genus_dict[GENUS]
pass_f1 = np.array([results_f1[results_name.index(mthd)] for mthd in pass_name])
pass_token = np.array([cumnum_token[results_name.index(mthd)] for mthd in pass_name])

L = len(results_f1[0])

batch_sizes = list(
    itertools.accumulate([16] + [BATCH_CONST * (2 ** i) for i in range(1, L)])
)
if DIVIDE == False:
    plot_single_genus(
        NAME, pass_name, batch_sizes, pass_f1, pass_token, results_error, INTPLT=INTPLT
    )
else:
    plot_single_genus(
        NAME + "_1",
        pass_name,
        batch_sizes[: len(batch_sizes) // 2],
        pass_f1[:, : pass_f1.shape[1] // 2],
        pass_token[:, : pass_token.shape[1] // 2],
        results_error[:, : results_error.shape[1] // 2],
        INTPLT=INTPLT,
    )
    plot_single_genus(
        NAME + "_2",
        pass_name,
        batch_sizes[len(batch_sizes) // 2 :],
        pass_f1[:, pass_f1.shape[1] // 2 :],
        pass_token[:, pass_token.shape[1] // 2 :],
        results_error[:, results_error.shape[1] // 2 :],
        INTPLT=INTPLT,
    )

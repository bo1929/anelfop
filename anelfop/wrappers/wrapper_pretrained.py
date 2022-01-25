import torch
import json
import os

from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report

from tqdm import tqdm
import numpy as np
import pickle as pkl


def get_embeddings(cfg, tknzd_sent, tags, pos, part="train"):

    pretrained_model = cfg["pretrained_model"]
    embedding_type = cfg["embedding_type"]

    tokenizer_ = AutoTokenizer.from_pretrained(
        pretrained_model, do_basic_tokenize=False
    )
    model = AutoModel.from_pretrained(pretrained_model)

    pretrained_tknzd = [
        tokenizer_(
            sent,
            return_tensors="pt",
            is_pretokenized=True,
            max_length=512,
            truncation=True,
        )
        for sent in tqdm(tknzd_sent)
    ]

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    main_dir = cfg["main_directory"]
    embeddings_dir = os.path.join(main_dir, "saved_embeddings", "")

    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)

    name_embeddings = (
        cfg["data_set"]["name"]
        + "_"
        + cfg["embedding_type"]
        + "_"
        + os.path.split(pretrained_model)[-1].replace(".", "")
        + "."
        + part
    )
    embedding_path = os.path.join(embeddings_dir, name_embeddings)

    if os.path.isfile(embedding_path):
        with open(embedding_path, "rb") as outfile:
            embeddings = pkl.load(outfile)
        print("Embeddings are loaded.")

    else:
        model.eval()
        with torch.no_grad():
            # Drop the [CLS] [SEP] token embeddings.
            if embedding_type == "ll":
                embeddings = [
                    model(**tknzdSent, output_hidden_states=True)[0][0, 1:-1, :]
                    for tknzdSent in tqdm(pretrained_tknzd)
                ]
            if embedding_type == "sl4l":
                embeddings = [
                    sum(model(**tknzdSent, output_hidden_states=True)[2][-4:])[
                        0, 1:-1, :
                    ]
                    for tknzdSent in tqdm(pretrained_tknzd)
                ]
            if embedding_type == "cl4l":
                embeddings = [
                    torch.cat(
                        [
                            layer[0, 1:-1, :]
                            for layer in model(**tknzdSent, output_hidden_states=True)[
                                2
                            ][-4:]
                        ],
                        dim=1,
                    )
                    for tknzdSent in tqdm(pretrained_tknzd)
                ]

        with open(embedding_path, "wb") as outfile:
            pkl.dump(embeddings, outfile)

    embedding_dim = embeddings[0].shape[1]

    # Averaging embeddings of the subwords into a single embedding.
    temp = []
    for i in tqdm(range(len(pretrained_tknzd))):
        temp_sent = [embeddings[i][0]]
        ##(1) n = 1
        for j in range(1, len(embeddings[i])):
            if (
                tokenizer_.decode([pretrained_tknzd[i]["input_ids"][0][j + 1]])[0]
                == "#"
            ):
                temp_sent[-1] = temp_sent[-1] + embeddings[i][j]
                ##(1) n = n + 1
            else:
                ##(1) if (
                ##(1)     tokenizer_.decode([pretrained_tknzd[i]["input_ids"][0][j - 1]])[0]
                ##(1)     == "#"
                ##(1) ):
                ##(1)     temp_sent[-1] = temp_sent[-1] / n
                ##(1)     n = 1
                temp_sent.append(embeddings[i][j])
        temp.append(temp_sent[:])

        # Truncation for tags and actual tokens, truncation can be done explicitly (rather than seperately).
        tknzd_sent[i] = tknzd_sent[i][: len(temp_sent)]
        tags[i] = tags[i][: len(temp_sent)]
        pos[i] = pos[i][: len(temp_sent)]

    embeddings = temp

    return embeddings, pretrained_tknzd, tknzd_sent, tags, pos
    irint("Tokenization -->")

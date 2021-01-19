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

    print("Tokenization -->")
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

    print("Extracting pretrained embeddings -->")
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
        print("embeddings are loaded")

    else:
        model.eval()
        with torch.no_grad():
            # we drop [CLS] [SEP] token embeddings                            ^o^
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

    print("\n\n\n\n----------------")
    print("last layer shape: ", embeddings[0].shape)
    print("embedding dimension: ", embedding_dim)
    print(
        "pretrained_tknzd shape: [0]  ",
        len(pretrained_tknzd),
        len(pretrained_tknzd[0]["input_ids"][0]),
    )
    print("embeddings shape: [0]   ", len(embeddings), len(embeddings[0]))
    print("tags shape: [0]         ", len(tags), len(tags[0]))
    print("tknzd_sent shape: [0]   ", len(tknzd_sent), len(tknzd_sent[0]))

    # averaging embeddings of the subwords into a single embedding
    temp = []
    for i in tqdm(range(len(pretrained_tknzd))):
        temp_sent = [embeddings[i][0]]
        #n = 1
        for j in range(1, len(embeddings[i])):
            if tokenizer_.decode([pretrained_tknzd[i]["input_ids"][0][j+1]])[0] == "#":
                temp_sent[-1] = temp_sent[-1] + embeddings[i][j]
                #n = n + 1
            else:
                """
                if (
                    tokenizer_.decode([pretrained_tknzd[i]["input_ids"][0][j - 1]])[0]
                    == "#"
                ):
                    temp_sent[-1] = temp_sent[-1] / n
                    n = 1
                """
                temp_sent.append(embeddings[i][j])
        temp.append(temp_sent[:])

        # Truncation for tags and actual tokens, truncation can be done explicitly (rather than seperately)
        tknzd_sent[i] = tknzd_sent[i][: len(temp_sent)]
        tags[i] = tags[i][: len(temp_sent)]
        pos[i] = pos[i][: len(temp_sent)]
        """
        if len(tags[i]) != len(temp_sent):
            print(tags[i])
            print(tknzd_sent[i])
            for j in range(len(pretrained_tknzd[i]["input_ids"][0])):
                print(tokenizer_.decode([pretrained_tknzd[i]["input_ids"][0][j]]))
            for j in range(len(embeddings[i])):
                print(tokenizer_.decode([pretrained_tknzd[i]["input_ids"][0][j]]))
        """
    embeddings = temp

    print("----------------\n\n\n\n")
    print("After summing embeddings of subwords; ")
    print(
        "pretrained_tknzd shape: [0]  ",
        len(pretrained_tknzd),
        len(pretrained_tknzd[0]["input_ids"][0]),
    )
    print("embeddings shape: [0]   ", len(embeddings), len(embeddings[0]))
    print("tags shape: [0]          ", len(tags), len(tags[0]))
    print("tknzd_sent shape: [0]    ", len(tknzd_sent), len(tknzd_sent[0]))

    return embeddings, pretrained_tknzd, tknzd_sent, tags, pos

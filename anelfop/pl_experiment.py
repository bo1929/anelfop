import argparse
import time as money

import wrappers.wrapper_CRF as crf_
import wrappers.wrapper_pretrained as pretrained_

import functions
import load_save

from seqeval.metrics import classification_report, f1_score

parser = argparse.ArgumentParser(prog="PROG")
parser.add_argument(
    "--config-path",
    required=True,
    type=str,
    help="Configuration file path to use, \
        seed, UMAP and HDBSCAN parameters",
)

args = parser.parse_args()

if __name__ == "__main__":
    config_path = args.config_path
    cfg = load_save.load_config_from(config_path, AL=False)
    random_seed = cfg["seed"]

    [tknzd_sent_train, tags_train, pos_train], [
        tknzd_sent_test,
        tags_test,
        pos_test,
    ] = load_save.load_data(cfg)

    (
        embeddings_train,
        pretrained_tknzd_train,
        tknzd_sent_train,
        y_train,
        pos_train,
    ) = pretrained_.get_embeddings(
        cfg, tknzd_sent_train, tags_train, pos_train, part="train"
    )

    (
        embeddings_test,
        pretrained_tknzd_test,
        tknzd_sent_test,
        y_test,
        pos_test,
    ) = pretrained_.get_embeddings(
        cfg, tknzd_sent_test, tags_test, pos_test, part="test"
    )

    embeddings_train_r, embeddings_test_r = functions.reduce_embeddings(
        cfg,
        embeddings_train,
        embeddings_test,
    )

    embedding_dim = embeddings_train_r[0][0].shape[0]

    load_save.write_ft_config(cfg)
    feature_cfg = load_save.load_ft_config(cfg)

    X_test = crf_.sent2features(
        feature_cfg,
        tknzd_sent_test,
        generator=cfg["generator"],
        embeddings=embeddings_test_r,
        pos=pos_test,
    )

    X_train = crf_.sent2features(
        feature_cfg,
        tknzd_sent_train,
        generator=cfg["generator"],
        embeddings=embeddings_train_r,
        pos=pos_train,
    )

    start = money.time()
    print("CRF training.\n")
    crf_trained = crf_.train_crf(cfg, X_train, y_train)

    print("CRF testing.\n")
    y_pred = crf_trained.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    end = money.time()

    load_save.save_crf_model(cfg, crf_trained, 0)
    load_save.save_results(
        cfg,
        [report, start - end, f1_score(y_test, y_pred)],
        [f1_score(y_test, y_pred)],
        [],
        [],
    )

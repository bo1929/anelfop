import re
import os
import numpy as np
import string
import json
from tqdm import tqdm
from nltk.corpus.reader import ConllCorpusReader
from nltk.util import LazyConcatenation, LazyMap
from pathlib import Path


class betterConllReader(ConllCorpusReader):
    def iob_sents(self, fileids=None, tagset=None):
        """
        :return: a list of lists of word/tag/IOB tuples
        :rtype: list(list)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.WORDS)

        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset)

        return LazyMap(get_iob_words, self._grids(fileids))

    def iob_words(self, fileids=None, tagset=None, column="ne"):
        """
        :return: a list of word/tag/IOB tuples
        :rtype: list(tuple)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.WORDS)

        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset, column)

        return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))

    def _get_iob_words(self, grid, tagset=None, column="ne"):
        return list(
            zip(
                self._get_column(grid, self._colmap["words"]),
                self._get_column(grid, self._colmap[column]),
            )
        )


sentences = []
tags = []
tokenized_sent = []
pos_tages = []

save_data_dir = Path(__file__).parent.absolute()
raw_data_dir = os.path.join(save_data_dir, "raw", "")
tokenized_data_dir = os.path.join(save_data_dir, "tokenized", "")

if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

if not os.path.exists(tokenized_data_dir):
    os.mkdir(tokenized_data_dir)

train = betterConllReader(raw_data_dir, "BC2GM_train.tsv", ["words", "ne"]).iob_sents()
test = betterConllReader(raw_data_dir, "BC2GM_test.tsv", ["words", "ne"]).iob_sents()

tokenized = []
for sents in [train, test]:
    token_c = []
    ne_c = []
    for sent in tqdm(sents):
        token_ = [token for token, ne in sent]
        ne_ = [ne for token, ne in sent]
        if token_:
            token_c.append(token_)
            ne_c.append(ne_)
    tokenized.append((token_c, ne_c))


with open(tokenized_data_dir + "BC2GM_train.tokenized", "w") as fp:  # serializing
    json.dump(tokenized[0][0], fp)
with open(tokenized_data_dir + "BC2GM_train.tags", "w") as fp:  # serializing
    json.dump(tokenized[0][1], fp)

with open(tokenized_data_dir + "BC2GM_test.tokenized", "w") as fp:  # serializing
    json.dump(tokenized[1][0], fp)
with open(tokenized_data_dir + "BC2GM_test.tags", "w") as fp:  # serializing
    json.dump(tokenized[1][1], fp)

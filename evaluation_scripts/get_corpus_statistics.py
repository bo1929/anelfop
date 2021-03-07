import os
import glob

import numpy as np

from tabulate import tabulate

if not os.path.exists("../evaluations/"):
    os.mkdir("../evaluations/")

if not os.path.exists("../evaluations/corpus_stats/"):
    os.mkdir("../evaluations/corpus_stats/")


# tags_corpus_paths = glob.glob(
#     "../datasets/tokenized/*.tags", recursive=True
# )

tokenized_corpus_paths = glob.glob(
    "../datasets/tokenized/*.tokenized", recursive=True
)

###################

details_tuple = []
for corpusTagPath in tokenized_corpus_paths:
    head, tail =  os.path.split(item[0])
    corpus_name, type = tail.split(".")[0].split("_")

    corpusTagPath = os.path.join(head, corpus_name+".tag")
    corpusTokenizedPath = os.path.join(head, corpus_name+".tokenized")
    details_tuple.append(
        [corpus_name, type, corpusTokenizedPath, corpusTagPath]
    )

def key_func(i):
    return lambda x: x[i]

details_tuple.sort(key=key_func(0))
for key, group1 in itertools.groupby(details_tuple, key_func(0)):
    for item in group1:

        with open(item[2], "rb") as openfile:
            tokenized_ = pickle.load(openfile)

        with open(item[3], "rb") as openfile:
            tags = pickle.load(openfile)

        

        table.append([item[1], item[3], item[4], item[5]] + f1avg)    
    
    table.sort(key=key_func[1])
    header_ = ["AL method", "pre-trained model", "embedding type", "embedding dimension"]+["f1-score "+str(i) for i in range(len(table[0])-4)]
    with open("../evaluations/active_tables/" + key + "_table_active_expt.tex", "w") as file1:
        file1.write(
            tabulate(
                table,
                headers=header_,
                tablefmt="latex",
            )
        )
    with open("../evaluations/active_tables/" + key + "_table_active_expt.md", "w") as file2:
        file2.write(
            tabulate(
                table,
                headers=header_,
                tablefmt="github",
            )
        )

tr_tags = tr_tags_test + tr_tags_train
eng_tags = eng_tags_test + eng_tags_train

tr_tags_flat = [tag for sent in tr_tags for tag in sent]
eng_tags_flat = [tag for sent in eng_tags for tag in sent]

from collections import Counter

count_tr_tags_flat = dict(Counter(tr_tags_flat))
count_eng_tags_flat = dict(Counter(eng_tags_flat))


print(len(tr_tags), len(eng_tags))
print(len(tr_tags_flat),len(eng_tags_flat))

b_count_en = {tag[1:]:count_eng_tags_flat[tag] for tag in count_eng_tags_flat.keys() if tag[0]=='B'}
i_count_en = {tag[1:]:count_eng_tags_flat[tag] for tag in count_eng_tags_flat.keys() if tag[0]=='I'}
count_eng_tags_flat = {tag:b_count_en[tag]+i_count_en[tag] for tag in b_count_en.keys()}

b_count_tr= {tag[2:]:count_tr_tags_flat[tag] for tag in count_tr_tags_flat.keys() if tag[0]=='B'}
i_count_tr= {tag[2:]:count_tr_tags_flat[tag] for tag in count_tr_tags_flat.keys() if tag[0]=='I'}
count_tr_tags_flat = {tag:b_count_tr[tag]+i_count_tr[tag] for tag in b_count_tr.keys()}

print(count_tr_tags_flat, count_eng_tags_flat)

tr_tags_PN = [[0 if tag=='O' else 1 for tag in sent] for sent in tr_tags]
eng_tags_PN =  [[0 if tag=='O' else 1 for tag in sent] for sent in eng_tags]

avg_SL_tr = mean([len(seq) for seq in tr_tags_PN])
avg_SL_eng = mean([len(seq) for seq in eng_tags_PN])

avg_Ptoken_tr = mean([sum(seq) for seq in tr_tags_PN])
avg_Ptoken_eng = mean([sum(seq) for seq in eng_tags_PN])

sum_Ptoken_tr = sum([sum(seq) for seq in tr_tags_PN])
sum_Ptoken_eng = sum([sum(seq) for seq in eng_tags_PN])


AC_Ptoken_tr = sum([1 if sum(seq)>=1 else 0 for seq in tr_tags_PN])/len(tr_tags_PN)
AC_Ptoken_eng = sum([1 if sum(seq)>=1 else 0 for seq in eng_tags_PN])/len(eng_tags_PN)

DAC_Ptoken_tr = sum([1 if sum(seq)>=2 else 0 for seq in tr_tags_PN])/len(tr_tags_PN)
DAC_Ptoken_eng = sum([1 if sum(seq)>=2 else 0 for seq in eng_tags_PN])/len(eng_tags_PN)

print(sum_Ptoken_tr, sum_Ptoken_eng)
print(avg_Ptoken_tr, avg_Ptoken_eng)
print(avg_SL_tr, avg_SL_eng)
print(avg_Ptoken_tr, avg_Ptoken_eng)
print(AC_Ptoken_tr, AC_Ptoken_eng)
print(DAC_Ptoken_tr, DAC_Ptoken_eng)

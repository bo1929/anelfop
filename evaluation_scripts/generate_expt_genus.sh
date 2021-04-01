#!/bin/bash

methods=("te" "tp" "ap" "tm")
corpora=("CONLL2003" "NCBI_disease" "s800" "BC5CDR")

for method_ in ${methods[@]}; do
    for corpus_ in ${corpora[@]}; do
        python plot_expt_genus.py "${corpus_}" "${method_}" "token-f1" "sentence-f1" "sentence-token"
        python plot_expt_genus.py "${corpus_}" "${method_}" "sentence-f1" "sentence-token"
        python plot_expt_genus.py "${corpus_}" "${method_}" "token-f1" "sentence-token"
        python plot_expt_genus.py "${corpus_}" "${method_}" "token-f1" "sentence-f1" 
    done
done

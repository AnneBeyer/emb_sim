#!/bin/bash

########## CONSTANTS

CORPUS1=$1
CORPUS2=$2

CORPUS_DIR=<DIR_CONTAINING_PREPROCESSED_CORPUS_FILES/>
EMBEDDING_DIR="embeddings/en_260MB/"
SIM_DIR="simulation/"$3
WORK_DIR=<PATH_TO_PPMI+SVD_IMPLEMENTATION>

OUT_DIR=${WORK_DIR}$3

########## FUNCTIONS

# simulation function
simulation(){

  # shuffle joint corpus
  shuf $1 > corpus.shuf.$2

  #split into two disjunct parts
  head -c 260MB corpus.shuf.$2 > corpus.1.$2.en.tmp
  head -n -1 corpus.1.$2.en.tmp > corpus.1.$2.en

  tail -c 260MB corpus.shuf.$2 > corpus.2.$2.en.tmp
  tail -n +2 corpus.2.$2.en.tmp > corpus.2.$2.en

  # train embeddings
  conda activate embeddings2
  cd ${WORK_DIR}
  ./corpus2svd.sh --thr 50 --win 5 --cds 0.75 --dim 100 --eig 0.0 $3/corpus.1.$2.en $3/1_$2 &>$3/1_$2.log
  ./corpus2svd.sh --thr 50 --win 5 --cds 0.75 --dim 100 --eig 0.0 $3/corpus.2.$2.en $3/2_$2 &>$3/2_$2.log

  # compare embeddings
  conda activate embeddings
  cd $3
  python3 ../mapping_correlation.py corpus.1.$2.en.emb corpus.2.$2.en.emb vocab.$2.txt

}  # end of simulation

########## MAIN

# use conda to switch environments (embedding requires py2.7, mapping script is written in py3.6)
source /<PATH_TO>/anaconda3/etc/profile.d/conda.sh

mkdir -p ${EMBEDDING_DIR}
mkdir -p ${OUT_DIR}

# check if original embeddings exists else create embeddings
if [[ ! -f ${EMBEDDING_DIR}${CORPUS1}.emb ]]; then
    cd ${WORK_DIR}
    conda activate embeddings2
    # train embedding
    ./corpus2svd.sh --thr 50 --win 5 --cds 0.75 --dim 100 --eig 0.0 ${CORPUS_DIR}${CORPUS1} ${OUT_DIR}/1 &>${OUT_DIR}/${CORPUS1}.log
    # move embedding file to embedding dir
    mv ${CORPUS_DIR}${CORPUS1}.emb ${EMBEDDING_DIR}${CORPUS1}.emb

fi
if [[ ! -f ${EMBEDDING_DIR}${CORPUS2}.emb ]]; then
    cd ${WORK_DIR}
    conda activate embeddings2
    # train embedding
    ./corpus2svd.sh --thr 50 --win 5 --cds 0.75 --dim 100 --eig 0.0 ${CORPUS_DIR}${CORPUS2} ${OUT_DIR}/2 &>${OUT_DIR}/${CORPUS2}.log

    # move original embedding files to embedding dir
    mv ${CORPUS_DIR}${CORPUS2}.emb ${EMBEDDING_DIR}${CORPUS2}.emb

fi

cd ${OUT_DIR}
# create output file
touch ${OUT_DIR}/$3.csv

# get original correlation
conda activate embeddings
python3 /mounts/work/beyera/universality-of-word-embeddings/scripts/mapping_correlation.py ${EMBEDDING_DIR}${CORPUS1}.emb ${EMBEDDING_DIR}${CORPUS2}.emb vocab.txt

# join corpora for simulation
cat ${CORPUS_DIR}${CORPUS1} ${CORPUS_DIR}${CORPUS2} >corpus.join

for i in {1..10}
do
    let "begin = 10 * ($i -1) + 1"
    let "end = 10 * $i"

    for ((j=begin;j<=end;j++)); do
        simulation corpus.join ${j} ${OUT_DIR} &
    done
    wait
done

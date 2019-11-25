#!/bin/bash

# adapt language and embed dir as required
LANG=en
SCRIPT=../../mapping_correlation.py
EMBED_DIR=../../embeddings/${LANG}_260MB

mkdir -p correlation/wiki1-wiki1
cd correlation/wiki1-wiki1
python3 ${SCRIPT} ${EMBED_DIR}/wiki.1.${LANG}.emb ${EMBED_DIR}/wiki.1.${LANG}.emb vocab.txt &>mapping.log

mkdir ../wiki1-wiki2
cd ../wiki1-wiki2
python3 ${SCRIPT} ${EMBED_DIR}/wiki.1.${LANG}.emb ${EMBED_DIR}/wiki.2.${LANG}.emb vocab.txt &>mapping.log

mkdir ../wiki-euro
cd ../wiki-euro
python3 ${SCRIPT} ${EMBED_DIR}/wiki.1.${LANG}.emb ${EMBED_DIR}/euro.${LANG}.emb vocab.txt &>mapping.log

mkdir  ../wiki-dgt
cd ../wiki-dgt
python3 ${SCRIPT} ${EMBED_DIR}/wiki.1.${LANG}.emb ${EMBED_DIR}/dgt.${LANG}.emb vocab.txt &>mapping.log

mkdir ../wiki-sub
cd ../wiki-sub
python3 ${SCRIPT} ${EMBED_DIR}/wiki.1.${LANG}.emb ${EMBED_DIR}/sub.${LANG}.emb vocab.txt &>mapping.log

mkdir ../wiki-med
cd ../wiki-med
python3 ${SCRIPT} ${EMBED_DIR}/wiki.1.${LANG}.emb ${EMBED_DIR}/med.${LANG}.emb vocab.txt &>mapping.log

cd ../..
python3 visualize.py correlation/


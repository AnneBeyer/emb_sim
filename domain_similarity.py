#!/usr/bin/env python3

"""
Calculate CCA measure for all specified corpus combinations
and create LaTeX table
"""

import logging
import argparse
from tabulate import tabulate
import numpy as np
from utils.UniversalityTests import UniversalityTests

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="Calculate CCA measure for all specified corpus combinations")
parser.add_argument('emb_path', type=str, help="Path to pre-trained embeddings")
parser.add_argument('work_dir', type=str, help="Where to store results and LaTeX output file")
args = parser.parse_args()

embeddings = ["books.en.emb", "dvd.en.emb", "electronics.en.emb", "kitchen.en.emb"] #"wiki.1.en.emb", "wiki.2.en.emb", "sub.en.emb", "dgt.en.emb", "euro.en.emb", "med.en.emb"]

algorithm = "gcca"  # "procrustes" "noise"

# calculate distances
headers = ['books', 'dvd', 'electronics', 'kitchen']#'wiki1', 'wiki2', 'sub', 'dgt', 'euro', 'med']
distances = [[], [], [], []] #, [], []]

for i, source_emb in enumerate(embeddings):
    distances[i].append(headers[i])
    for j, target_emb in enumerate(embeddings):

        # load embeddings
        embedding_tests = UniversalityTests(args.emb_path+source_emb, args.emb_path+target_emb, args.work_dir+headers[i]+"_"+headers[j]+".vocab.txt")

        # map spaces
        embedding_tests.map_spaces(algorithm, src_mapped_embed="mapped_src_" + source_emb,
                                   trg_mapped_embed="mapped_trg_" + target_emb)

        # calculate CCA measure
        corr = embedding_tests.get_embedding_correlations()
        cca_measure = np.mean(corr)
        distances[i].append(cca_measure)

latex = tabulate(distances, headers=[''] + headers, floatfmt='0.2f',
                          tablefmt='latex')
print(latex)
with open(args.work_dir+'similarity_table.txt', 'w') as out:
    out.write(latex)

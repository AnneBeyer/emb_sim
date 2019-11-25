#!/usr/bin/env python3

'''
Compute CCA measure for two pre-trained embedding spaces
'''

import argparse
import logging
import os
import numpy as np
import csv
import threading
from utils.UniversalityTests import UniversalityTests

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="Compute correlation after mapping two embedding spaces")
parser.add_argument('src_emb', type=str, help="source embedding")
parser.add_argument('trg_emb', type=str, help="target embedding")
parser.add_argument('vocab', type=str, help="file for loading/saving shared vocabulary")
parser.add_argument('dict', nargs='?', default=None, type=str, help="dictionary for extracting shared vocabulary in cross-lingual comparison")

args = parser.parse_args()

# csv file will contain dimensions-wise correlations for visualization
csv_file = os.path.basename(os.getcwd()) + ".csv"
csv_writer_lock = threading.Lock()
# log file will contain CCA measure scores before and after mapping embedding spaces
logfile = os.path.basename(os.getcwd()) + ".log"

algorithm = "gcca"  # "procrustes" "noise"

with open(csv_file, mode='a') as corr_file:
    corr_writer = csv.writer(corr_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # load embeddings
    embedding_tests = UniversalityTests(args.src_emb, args.trg_emb, args.vocab, dictionary=args.dict)

    # calculate dimension wise correlations
    corr = embedding_tests.get_embedding_correlations()
    # convert to CCA measure
    cca_measure_pre = np.mean(corr)

    # map spaces
    embedding_tests.map_spaces(algorithm, src_mapped_embed="mapped_src_" + os.path.basename(args.src_emb),
                               trg_mapped_embed="mapped_trg_" + os.path.basename(args.trg_emb))

    # calculate dimension wise correlations
    corr_post = embedding_tests.get_embedding_correlations()
    # convert to CCA measure
    cca_measure_post = np.mean(corr_post)

    with open(logfile, 'w') as out:
        out.write("Pre-map scores\nCCA measure: {}\nPost-map scores\nCCA measure: {}\n".format(cca_measure_pre, cca_measure_post))

    with csv_writer_lock:
        if algorithm == "gcca":
            # gcca implementation returns correlations in ascending order
            corr_post = np.flip(corr_post)
        corr_writer.writerow(corr_post)

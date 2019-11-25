#!/usr/bin/env python3

import os
import logging
import random
import numpy as np
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors  # for loading pre-trained word vectors
from sklearn.cross_decomposition import CCA
from utils.gcca import GCCA  # gcca implementation (faster than cca)
from utils.noise_aware import noise_aware  # procrustes adaptation by Lubin et al. (2019)
# from tabulate import tabulate # for creating LaTeX tables
import matplotlib
matplotlib.use('Agg') # needed to create plots on server
import matplotlib.pyplot as plt


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

EMBEDSIZE = 100

class UniversalityTests:
    """
    Main class for loading, mappig and calculating correlations of embedding spaces
    """
    def __init__(self, src_embed, trg_embed, vocab_file, dictionary=None, norm=False):

        self.norm = norm
        # load pre-trained word vectors
        logging.debug("Loading embeddings")
        self.model_src = KeyedVectors.load(src_embed)
        self.model_trg = KeyedVectors.load(trg_embed)

        if self.norm:
            # create l2 normalized vectors (replacing vectors with norm)
            self.model_src.init_sims(replace=True)
            self.model_trg.init_sims(replace=True)

        # get shared vocab
        self.shared_vocab = self.get_vocab(vocab_file, dictionary)
        self.shared_vocab_src, self.shared_vocab_trg = zip(*self.shared_vocab)

    def get_vocab(self, vocab_file, dictionary):
        shared_vocab = []
        src_trg = defaultdict(set)
        trg_src = defaultdict(set)

        if os.path.isfile(vocab_file):
            # load shared vocab (already shuffled)
            logging.debug("Loading shared vocabulary")
            with open(vocab_file, "r", encoding="utf-8") as vf:
                for line in vf:
                    pair = line.strip().split()
                    src_trg[pair[0]].add(pair[1])
                    trg_src[pair[1]].add(pair[0])
                    shared_vocab.append(tuple(pair))
        else:
            # get shared vocabulary and store it in file for reproducibility
            if dictionary:
                logging.debug("Extracting translations")
                translations = set()
                with open(dictionary) as d:
                    for line in d:
                        pair = line.strip().split()
                        translations.add((pair[0], pair[1]))
                        src_trg[pair[0]].add(pair[1])
                        trg_src[pair[1]].add(pair[0])
                logging.debug("Extracting shared bilingual vocabulary")
                shared_vocab = [pair for pair in translations if
                                     pair[0] in self.model_src.vocab and pair[1] in self.model_trg.vocab]
            else:
                logging.debug("Extracting shared monolingual vocabulary")
                shared_vocab = [(token, token) for token in self.model_src.vocab if
                                                       token in self.model_trg.vocab]

            # save shuffled shared vocabulary
            random.shuffle(shared_vocab)
            with open(vocab_file, "w", encoding="utf-8") as vf:
                for entry in shared_vocab:
                    vf.write("{} {}\n".format(entry[0], entry[1]))

        return shared_vocab

    def map_spaces(self, algo, src_mapped_embed=None, trg_mapped_embed=None):

        # (There may be duplicates in self.shared_vocab_src and/or self.shared_vocab_trg,
        # swap_vocab can be used to only inspect one-to-one translations)
        src_embed = self.model_src[self.shared_vocab_src]
        trg_embed = self.model_trg[self.shared_vocab_trg]

        os.makedirs(algo, exist_ok=True)

        if algo == "procrustes":
            logging.info("Calculating Rotation Matrix (Procrustes Problem) and applying it to first embedding")
            #ortho, _ = orthogonal_procrustes(src_embed, trg_embed)
            # does the same as
            u, _, vt = np.linalg.svd(trg_embed.T.dot(src_embed))
            w = vt.T.dot(u.T)
            self.model_src.vectors.dot(w, out=self.model_src.vectors)

        elif algo == "noise":
            logging.info("Calculating Rotation Matrix with noise aware algorithm and applying it to first embedding")
            transform_matrix, alpha, clean_indices, noisy_indices = noise_aware(src_embed,trg_embed)
            #write cleaned vocab to file
            with open("vocab.clean.txt", 'w') as v:
                for src, trg in np.asarray(self.shared_vocab)[clean_indices]:
                    v.write("{}\t{}\n".format(src, trg))
            self.model_src.vectors.dot(transform_matrix, out=self.model_src.vectors)
            logging.info("Percentage of clean indices: {}".format(alpha))

        elif algo == "cca":
            logging.info("Calculating Mapping based on CCA and applying it to both embeddings")
            cca = CCA(n_components=100, max_iter=5000)
            cca.fit(src_embed, trg_embed)
            self.model_src.vectors, self.model_trg.vectors = cca.transform(self.model_src.vectors, self.model_trg.vectors)

        elif algo == "gcca":
            logging.info("Calculating Mapping based on GCCA and applying it to both embeddings")
            gcca = GCCA()
            gcca.fit([src_embed, trg_embed])
            transform_l = gcca.transform_as_list((self.model_src.vectors, self.model_trg.vectors))
            # gcca computes positive and negative correlations (eigenvalues), sorted in ascending order.
            # We are only interested in the positive portion
            self.model_src.vectors = transform_l[0][:,100:]
            self.model_trg.vectors = transform_l[1][:,100:]

        # save transformed model(s)
        if src_mapped_embed:
            self.model_src.save(os.path.join(algo, src_mapped_embed))
        if trg_mapped_embed:
            self.model_trg.save(os.path.join(algo, trg_mapped_embed))

    def get_embedding_correlations(self):
        logging.debug("Calculating correlation matrix")
        src = self.model_src[self.shared_vocab_src]
        trg = self.model_trg[self.shared_vocab_trg]

        corr_matrix = np.corrcoef(src, trg, rowvar=False)
        # corr_matrix is 2*dim x 2*dim matrix, we are only interested in the corr between different embeddings,
        # i.e. upper right or lower left dim x dim matrix (they are the same, so we just pick one here)
        corr_matrix = corr_matrix[100:, :100]

        # d_frobenius_corrcoef = norm(corr_matrix) # how to interpret results?
        # rather inspect diagonal (dimension wise correlations)
        diag = np.diag(corr_matrix)
        return diag
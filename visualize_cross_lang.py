#!/usr/bin/env python3

"""
Visualize cross-lingual simulation results across domains.
Expects sub folders containing domain folders with dimension wise correlations in one line in a csv file
Creates cross_ling.png visualizing the similarities between languages across domains in a bar plot
"""

import os
import logging
from collections import defaultdict

import numpy as np
import csv
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

scores_dict = defaultdict(lambda: defaultdict(float))
for lang in ["en-en", "en-de", "en-es", "en-cs"]:
    wiki_path = os.path.abspath(os.path.join(lang, "wiki/gcca"))
    if os.path.isdir(wiki_path):
        for sub_dir in os.listdir(wiki_path):
            sub_path = os.path.join(wiki_path,sub_dir)
            if os.path.isdir(sub_path):
                for file in os.listdir(sub_path):
                    if file.endswith(".csv"):
                        with open(os.path.join(sub_path, file)) as f:
                            csvReader = csv.reader(f)
                            name = os.path.splitext(file)[0]
                            for row in csvReader:
                                orig = np.mean(row)
                                scores_dict[lang][name] = orig
                                print("{}: {}: {}\n".format(lang, name, round(orig,2)))
                                break

# data to plot
n_groups = 5

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8

colors = {"en-en":'purple', "en-de":'g', "en-es":'b', "en-cs":"orange"}
add = 0
for lang in scores_dict:
    scores = scores_dict[lang]
    plt.bar(index + add, (scores['wiki1-wiki2'], scores['wiki-sub'], scores['wiki-dgt'], scores['wiki-euro'], scores['wiki-med']), bar_width, alpha=opacity, color=colors[lang], label=lang)
    add += bar_width

plt.xlabel('Cross-lingual corpora comparisons')
plt.ylabel('Scores')
plt.title('Cross-lingual comparison')
plt.xticks(index + 1.5*bar_width, ('wiki1-wiki2', 'wiki-sub', 'wiki-dgt', 'wiki-euro', 'wiki-med'))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cross-ling.png")


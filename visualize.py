#!/usr/bin/env python3

"""
Visualize simulation results.
Expects subfolders containing csv files with simulation results
First line is original data distribution, following lines are simulation results
Creates corr.png visualizing dimension-wise correlation among all different comparisons and
one file per csv file visualizing the density distribution of the simulation results and the original datapoint
"""

import os
import sys
import logging
import numpy as np
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def plot_simulation(simulations, orig, name):
    """
    Create density plot of simulation results and save is as <name>.png in current directory
    :param simulations: simulated data points
    :param orig: original data point
    :param name: which corpora were compared
    """""
    plt.xlabel('dimension')
    plt.ylabel('correlation')
    fig, ax = plt.subplots()
    # Draw the density plot
    sns.distplot(simulations, hist=False, kde=True,
                 kde_kws={'linewidth': 2}, ax=ax)

    ax.plot(orig, 0, 'rx')
    ax.set_title(name)
    # remove ticks on y axis
    #ax.axes.get_yaxis().set_ticks([])
    # use same scale for all plots
    #plt.xticks(np.arange(0.2, 1, 0.05))
    minim = round(min(min(simulations), orig) - 0.0001, 4)
    maxim = round(max(max(simulations), orig) + 0.0001, 4)

    plt.xticks(np.arange(minim, maxim, 0.00001))
    # rotate labels if overlap
    ax.tick_params(axis='x', rotation=70)
    fig.tight_layout()
    plt.savefig(name + ".png")


def plot_correlation(path, corr_dict):
    """
    Plot correlation as function of embedding dimension for given
    """
    x = np.arange(100)
    fig, ax = plt.subplots()
    ax.set_xlabel('dimension')
    ax.set_ylabel('correlation')

    # hard coded to assure color-compatibility (uncomment desired setting)

    #plt.plot(x, [float(v) for v in corr_dict["wiki-wiki"]], color="purple", linestyle='solid', label="wiki-wiki")
    plt.plot(x, [float(v) for v in corr_dict["wiki1-wiki1"]], color="purple", linestyle='solid', label="wiki1-wiki1")
    plt.plot(x, [float(v) for v in corr_dict["wiki1-wiki2"]], color="green", linestyle='solid', label="wiki1-wiki2")
    plt.plot(x, [float(v) for v in corr_dict["wiki-sub"]], color="red", linestyle='solid', label="wiki-sub")
    plt.plot(x, [float(v) for v in corr_dict["wiki-euro"]], color="blue", linestyle='solid', label="wiki-euro")
    plt.plot(x, [float(v) for v in corr_dict["wiki-dgt"]], color="orange", linestyle='solid', label="wiki-dgt")
    plt.plot(x, [float(v) for v in corr_dict["wiki-med"]], color="c", linestyle='solid', label="wiki-med")

    # plt.plot(x, [float(v) for v in corr_dict["en-en"]], color="red", linestyle='solid', label="en-en")
    # plt.plot(x, [float(v) for v in corr_dict["en-de"]], color="blue", linestyle='solid', label="en-de")
    # plt.plot(x, [float(v) for v in corr_dict["en-de_en"]], color="blue", linestyle='dotted', label="en-de_en")
    # plt.plot(x, [float(v) for v in corr_dict["en-es"]], color="green", linestyle='solid', label="en-es")
    # plt.plot(x, [float(v) for v in corr_dict["en-cs"]], color="orange", linestyle='solid', label="en-cs")

    # plt.plot(x, [float(v) for v in corr_dict["wiki_en-wiki_en"]], color="purple", linestyle='solid', label="wiki_en-wiki_en")
    # plt.plot(x, [float(v) for v in corr_dict["wiki_en-wiki_de"]], color="green", linestyle='solid', label="wiki_en-wiki_de")
    # plt.plot(x, [float(v) for v in corr_dict["wiki_en-sub_de"]], color="red", linestyle='solid', label="wiki_en-sub_de")
    # plt.plot(x, [float(v) for v in corr_dict["wiki_en-euro_de"]], color="blue", linestyle='solid', label="wiki_en-euro_de")
    # plt.plot(x, [float(v) for v in corr_dict["wiki_en-dgt_de"]], color="orange", linestyle='solid', label="wiki_en-dgt_de")
    # plt.plot(x, [float(v) for v in corr_dict["wiki_en-med_de"]], color="c", linestyle='solid', label="wiki_en-med_de")

    # plt.plot(x, [float(v) for v in corr_dict["book-dvd"]], color="purple", linestyle='solid', label="book-dvd")
    # plt.plot(x, [float(v) for v in corr_dict["book-electronics"]], color="green", linestyle='solid', label="book-electronics")
    # plt.plot(x, [float(v) for v in corr_dict["book-kitchen"]], color="red", linestyle='solid', label="book-kitchen")
    # plt.plot(x, [float(v) for v in corr_dict["dvd-electronics"]], color="blue", linestyle='solid', label="dvd-electronics")
    # plt.plot(x, [float(v) for v in corr_dict["dvd-kitchen"]], color="orange", linestyle='solid', label="dvd-kitchen")
    # plt.plot(x, [float(v) for v in corr_dict["electronics-kitchen"]], color="c", linestyle='solid', label="electronics-kitchen")

    ax.legend(loc="upper left", bbox_to_anchor=(1,1))
    fig.tight_layout()
    plt.savefig(path + "/corr.png")


corr_dict = {}
# expects csv files in corresponding sub folders
# use commented version if csv files are in one folder
parent = os.path.abspath(sys.argv[1])
for d in os.listdir(parent):
    dir = os.path.join(parent,d)
    if os.path.isdir(dir):
        for file in os.listdir(dir):
            if file.endswith(".csv"):
                with open(os.path.join(dir,file)) as f:
# for file in os.listdir("."):
#     if file.endswith(".csv"):
#         with open(file) as f:
                    csvReader = csv.reader(f)
                    name = os.path.splitext(file)[0]
                    scores = []
                    is_first = True
                    for row in csvReader:
                        # first line contains original correlations
                        if is_first:
                            orig = np.mean([float(i) for i in row])
                            corr_dict[name] = row
                            print("{}: {}\n".format(name, round(orig,2)))
                            is_first = False
                        # next lines (if present) are simulation results
                        else:
                            scores.append(np.mean([float(i) for i in row]))
                # uncomment to visualize simulation results
                # plot_simulation(scores, orig, name)

plot_correlation(parent, corr_dict)


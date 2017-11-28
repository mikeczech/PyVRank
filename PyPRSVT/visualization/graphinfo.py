import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import networkx as nx
from PyPRSVT.preprocessing.graphs import EdgeType
import numpy as np
import functools
from matplotlib.colors import Normalize

def generate_node_number_hist(graphs, out):
    node_numbers = [g.number_of_nodes() for g in graphs]
    # the histogram of the data
    plt.figure(figsize=(17, 12))
    matplotlib.rc('xtick', labelsize=45)
    matplotlib.rc('ytick', labelsize=45)
    plt.tick_params(
    which='both',
    pad=20)      # both major and minor ticks are affected
    plt.tick_params(
            axis='x',
            which='both',
            pad=40)      # both major and minor ticks are affected
    hist = plt.hist(node_numbers, facecolor='blue', bins=np.logspace(1.0, 4.0, 90), edgecolor='none', normed=True)
    plt.gca().set_xscale("log")
    plt.savefig(out, format='PDF')


def draw_heatmap(out):
    plt.figure(figsize=(7, 7))
    # scores_memsafety = [[0.780, 0.776, 0.776, 0.771],
    #           [0.771, 0.773, 0.769, 0.766],
    #           [0.767, 0.764, 0.766, 0.753],
    #           [0.734, 0.750, 0.745, 0.738]]
    time_memsafety = [[6.31, 6.34, 6.35, 6.31],
              [6.64, 6.60, 6.68, 6.60],
              [8.35, 7.46, 7.45, 7.24],
              [19.0, 14.2, 11.5, 9.80]]
    time_termination = [[5.43, 5.05, 5.07, 5.08],
              [8.82, 5.79, 5.45, 5.48],
              [16.9, 9.06, 6.82, 6.59],
              [19.87, 19.8, 15.9, 13.0]]
    time_safety = [[11.6, 11.7, 11.36, 11.0],
              [15.2, 13.6, 13.0, 12.2],
              [25.1, 16.7, 14.7, 13.0],
              [70.9, 66.2, 42.6, 33.9]]
    gram_gen = [[1040, 1744, 2415, 3002],
                   [427, 726, 1021, 1298],
                   [257, 428, 600, 761],
                   [81, 134, 184, 237]]
    # scores_termination = [[0.680, 0.705, 0.687, 0.677],
    #           [0.667, 0.709, 0.685, 0.677],
    #           [0.634, 0.667, 0.674, 0.666],
    #           [0.554, 0.631, 0.624, 0.643]]
    # scores_safety = [[0.598, 0.607, 0.610, 0.611],
    #           [0.603, 0.610, 0.612, 0.612],
    #           [0.604, 0.605, 0.610, 0.613],
    #           [0.592, 0.598, 0.601, 0.603]]

    # scores = functools.reduce(np.add, [np.array(scores_memsafety), np.array(scores_termination), np.array(scores_safety)])
    # scores = np.divide(scores, 3)
    scores = functools.reduce(np.add, [np.array(time_memsafety), np.array(time_termination), np.array(time_safety)])
    scores = np.divide(scores, 3)

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    plt.tick_params(
            which='both',
            pad=10)      # both major and minor ticks are affected
    plt.tick_params(
            axis='x',
            which='both',
            pad=10)      # both major and minor ticks are affected
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
    #            norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.imshow(gram_gen, interpolation='nearest', cmap=plt.cm.hot_r)
    plt.xlabel('D', fontsize=10)
    plt.ylabel('h', fontsize=10, rotation=0)
    plt.colorbar()
    plt.xticks(np.arange(4), [0, 1, 2, 5], rotation=0)
    plt.yticks(np.arange(4), [5, 2, 1, 0])
    plt.title('Time for matrix generation', fontsize=10)
    plt.savefig(out, format='PDF')
    #plt.show()


def generate_edge_number_hist(graphs, out):
    edge_numbers_se = []
    edge_numbers_cf = []
    edge_numbers_ce = []
    edge_numbers_de = []
    for g in graphs:
        se_count = 0
        cf_count = 0
        ce_count = 0
        de_count = 0
        edge_types = nx.get_edge_attributes(g, 'type')
        for e in g.edges_iter(keys=True):
            if edge_types[e] == EdgeType(1):
                de_count += 1
            elif edge_types[e] == EdgeType(2):
                ce_count += 1
            elif edge_types[e] == EdgeType(3):
                cf_count += 1
            elif edge_types[e] == EdgeType(4):
                se_count += 1
        edge_numbers_se.append(se_count)
        edge_numbers_cf.append(cf_count)
        edge_numbers_ce.append(ce_count)
        edge_numbers_de.append(de_count)

    plt.figure(figsize=(17, 12))
    matplotlib.rc('xtick', labelsize=45)
    matplotlib.rc('ytick', labelsize=45)
    plt.tick_params(
            which='both',
            pad=20)      # both major and minor ticks are affected
    plt.tick_params(
            axis='x',
            which='both',
            pad=40)      # both major and minor ticks are affected
    plt.hist(edge_numbers_cf, facecolor='green', bins=np.logspace(1.0, 3.0, 90), alpha=1.0, normed=True, edgecolor='none', label='CFE')
    plt.hist(edge_numbers_se, facecolor='blue', bins=np.logspace(1.0, 3.0, 90), alpha=1.0, normed=True, edgecolor='none', label='SE')
    #plt.hist(edge_numbers_de, facecolor='green', bins=np.logspace(0.1, 3.0, 90), alpha=1.0, normed=True, edgecolor='none', label='DE')
    #plt.hist(edge_numbers_ce, facecolor='blue', bins=np.logspace(0.1, 3.0, 90), alpha=1.0, normed=True, edgecolor='none', label='CE')
    plt.legend(loc='upper right', prop={'size':45})
    plt.gca().set_xscale("log")
    plt.savefig(out, format='PDF')


def generate_node_depth_hist(graphs, out):
    depths_list = []
    for g in graphs:
        node_depths = nx.get_node_attributes(g, 'depth')
        depths_list.extend([node_depths[n] for n in g.nodes_iter()])
    plt.figure(figsize=(17, 12))
    matplotlib.rc('xtick', labelsize=45)
    matplotlib.rc('ytick', labelsize=45)
    plt.tick_params(
            which='both',
            pad=20)      # both major and minor ticks are affected
    plt.tick_params(
            axis='x',
            which='both',
            pad=40)      # both major and minor ticks are affected
    plt.hist(depths_list, facecolor='blue', bins=10, edgecolor='none', normed=True)
    plt.savefig(out, format='PDF')


def generate_node_degree_hist(graphs, out):
    degree_list = []
    for g in graphs:
        degree_list.extend([g.in_degree(n) for n in g.nodes_iter()])
    plt.figure(figsize=(17, 12))
    matplotlib.rc('xtick', labelsize=45)
    matplotlib.rc('ytick', labelsize=45)
    plt.tick_params(
            which='both',
            pad=20)      # both major and minor ticks are affected
    plt.tick_params(
            axis='x',
            which='both',
            pad=40)      # both major and minor ticks are affected
    plt.hist(degree_list, facecolor='blue', bins=200, edgecolor='none', normed=True)
    plt.savefig(out, format='PDF')

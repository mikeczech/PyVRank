from PyPRSVT.ranking import rpc, distance_metrics
from PyPRSVT.gk import GK_WL as gk
from PyPRSVT.preprocessing.graphs import EdgeType
from PyPRSVT.preprocessing.ranking import Ranking
from ast import literal_eval
import pandas as pd
from sklearn import cross_validation
from os.path import isfile, join, exists
import argparse
import numpy as np
import networkx as nx
import random
import re
import itertools
import time
from tqdm import tqdm


def dump_gram(graph_paths, types, h_set, D_set, out_dir):
    graphs = []
    for p in graph_paths:
        if not isfile(p):
            raise ValueError('Graph {} not found.'.format(p))
        g = nx.read_gpickle(p)
        graphs.append(g)

    ret = {}
    h_D_product = list(itertools.product(h_set, D_set))
    matrix_computation_times = {}
    for i, (h, D) in enumerate(h_D_product):
        output_path = join(out_dir, 'K_h_{}_D_{}.gram'.format(h, D))
        ret[h, D] = output_path + '.npy'
        if not isfile(output_path + '.npy'):
            print('Computing normalized gram matrix for h={} and D={} ({} of {})'.format(h, D, i+1, len(h_D_product)), flush=True)
        else:
            print('{} already exists. Skipping...'.format(output_path + '.npy'))
            continue
        start_time = time.time()
        kernel = gk.GK_WL()
        K = kernel.compare_list_normalized(graphs, types, h, D)
        matrix_computation_times[h, D] = time.time() - start_time
        # saving matrix
        np.save(output_path, K)
    return ret, matrix_computation_times


def read_data(path):
    df = pd.DataFrame.from_csv(path)
    tools = []
    with open(path + '.tools') as f:
        tools.extend([x.strip() for x in f.readline().split(',')])
    return df, tools


def start_experiments(gram_paths, y, tools, h_set, D_set, folds=10):
    start_time = time.time()
    spearman = distance_metrics.SpearmansRankCorrelation(tools)
    scores = []
    loo = cross_validation.KFold(len(y), folds, shuffle=True, random_state=random.randint(0, 100))
    for train_index, test_index in tqdm(list(loo)):
        y_train, y_test = y[train_index], y[test_index]
        clf = rpc.RPC(tools, spearman)
        clf.gram_fit(h_set, D_set, [100], gram_paths, train_index, y_train)
        score = clf.score(gram_paths, test_index, train_index, y_test)
        scores.append(score)
    print('--- {} seconds ---'.format(time.time() - start_time), flush=True)
    return np.mean(scores), np.std(scores), clf.params


def write_statistics(file, gram_times):
    with open(file, 'w') as f:
        total_time = 0
        for (h, D), t in gram_times.items():
            f.write('h {}, D {}: {} seconds\n'.format(h, D, t))
            total_time += t
        f.write('Total time: {} seconds\n'.format(total_time))


def main():
    parser = argparse.ArgumentParser(description='Label Ranking')
    parser.add_argument('-g', '--dump_gram', type=str, required=False)
    parser.add_argument('--gram_dir', type=str, required=False)
    parser.add_argument('-t', '--types', type=int, nargs='+', required=False)
    parser.add_argument('--h_set', type=int, nargs='+', required=False)
    parser.add_argument('--D_set', type=int, nargs='+', required=False)
    parser.add_argument('-e', '--experiments', type=str, required=False)
    args = parser.parse_args()

    # Precompute gram matrices
    if all([args.dump_gram, args.gram_dir, args.types, args.h_set, args.D_set]):

        print('Write gram matrices of {} to {}.'.format(args.dump_gram, args.gram_dir), flush=True)

        if not all([EdgeType(t) in EdgeType for t in args.types]):
            raise ValueError('Unknown edge type detected')
        if not exists(args.gram_dir):
            raise ValueError('Given directory does not exist')
        df, tools = read_data(args.dump_gram)
        graph_series = df['graph_representation']
        types = [EdgeType(t) for t in args.types]
        gram_paths, times = dump_gram(graph_series.tolist(), types, args.h_set, args.D_set, args.gram_dir)
        with open(join(args.gram_dir, 'all.txt'), 'w') as f:
            for (h, D), v in gram_paths.items():
                f.write('{},{},{}\n'.format(h, D, v))
        write_statistics(join(args.gram_dir, 'gram.statistics'), times)


    # Perform experiments
    elif all([args.experiments, args.gram_dir, args.h_set, args.D_set]):
        if not exists(args.gram_dir):
            raise ValueError('Given directory does not exist')
        gram_paths = {}
        with open(join(args.gram_dir, 'all.txt')) as f:
            for l in f:
                m = re.match(r"([0-9]+),([0-9]+),(.+)\n", l)
                if m is not None:
                    h, D, path = int(m.group(1)), int(m.group(2)), m.group(3)
                    gram_paths[h, D] = path
                else:
                    raise ValueError('Invalid all.txt file?')
        df, tools = read_data(args.experiments)
        y = np.array([Ranking(literal_eval(r)) for r in df['ranking'].tolist()])
        mean, std, params = start_experiments(gram_paths, y, tools, args.h_set, args.D_set)
        print('Mean: {}, Std: {}'.format(mean, std))
        print(params)

    # Wrong arguments, therefore print usage
    else:
        parser.print_usage()
        quit()


if __name__ == '__main__':
    main()



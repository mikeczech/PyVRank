from PyPRSVT.preprocessing.competition import svcomp15
from PyPRSVT.preprocessing import ranking
from PyPRSVT.preprocessing.verifolio import features as vf
from PyPRSVT.preprocessing.cfa.generation import LabeledDiGraphGen
import PyPRSVT.gk.GK_WL as gk
import argparse
import pandas as pd
import networkx as nx
import numpy as np
import os
from os.path import join, isfile


def c_or_i(f):
    return f.endswith('.i') or f.endswith('.c')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create RPC Input from SVCOMP data')
    parser.add_argument('-c', '--category', type=str, nargs='+', required=False)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-d', '--dir', type=str, required=False)
    parser.add_argument('--cpachecker', type=str, required=False)
    parser.add_argument('--csv', type=str, required=False)
    parser.add_argument('--rankings', action='store_true', required=False)
    parser.add_argument('--features', action='store_true', required=False)
    parser.add_argument('--cfg', action='store_true', required=False)
    parser.add_argument('--gram', action='store_true', required=False)
    args = parser.parse_args()
    if not any([args.rankings, args.features, args.cfg, args.gram]):
        parser.print_usage()
        quit()

    # Write data set with tool preferences to output
    if all([args.rankings, args.category, args.dir]):
        for c in args.category:
            print('Processing category ' + c)
            results = svcomp15.read_category(args.dir, c)
            df, tools = ranking.create_benchmark_ranking_df(results, svcomp15.compare_results)
            df.to_csv(args.output + "." + c + ".csv")
            with open(args.output + '.' + c + '.tools', 'w') as f:
                f.write(",".join(tools))

    # Write data set with verifolio features to output
    elif all([args.features, args.csv, args.output]):
        source_files = []
        for csv in args.csv:
            df = pd.DataFrame.from_csv(csv)
            for (index, _) in df.iterrows():
                source_files.append(index)
        ret = vf.create_feature_df(set(source_files))
        ret.to_csv(args.output)

    elif all([args.cfg, args.csv, args.category, args.dir, args.cpachecker, args.output]):
        for c in args.category:
            source_files = []
            df = pd.DataFrame.from_csv(args.csv + '.' + c + '.csv')
            for (index, _) in df.iterrows():
                source_files.append(index)
            cfg = LabeledDiGraphGen(args.cpachecker)
            out_dir = join(args.dir, c)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            ret = cfg.create_digraph_df(source_files, out_dir)
            ret.to_csv(args.output + '.' + c + '.csv')

    elif all([args.gram, args.csv, args.output]):
        for csv in args.csv:
            graphs_df = pd.DataFrame.from_csv(csv)
            graph_list = []
            for _, row in graphs_df.iterrows():
                print('Processing ' + row.iloc[0])
                nx_digraph = nx.read_dot(row.iloc[0])
                graph_list.append(nx_digraph)
            X = gk.compare_list_normalized(graph_list, h=2)
            np.save(args.output, X)

    # Wrong arguments, therefore print usage
    else:
        parser.print_usage()
        quit()

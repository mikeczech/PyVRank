import argparse
import os
import pandas as pd
from PyPRSVT.preprocessing import ranking, svcomp15, graphs, verifolio
import time
import numpy as np
from PyPRSVT.ranking.distance_metrics import SpearmansRankCorrelation
from PyPRSVT.ranking.rpc import Ranking
from PyPRSVT.visualization import graphinfo
import networkx as nx


def c_or_i(f):
    return f.endswith('.i') or f.endswith('.c')


def extract_ranking_df(xml_dir, category, max_size):
    results = svcomp15.read_category(xml_dir, category, max_size)
    return ranking.create_ranking_df(results, svcomp15.compare_results)


def extract_feature_df(xml_dir, category, max_size):
    ranking_df, tools = extract_ranking_df(xml_dir, category, max_size)
    verification_tasks = []
    for (index, _) in ranking_df.iterrows():
        verification_tasks.append(index)
    feature_df = verifolio.create_feature_df(verification_tasks)
    ret_df = pd.concat([ranking_df, feature_df], axis=1)
    ret_df.dropna(inplace=True)
    return ret_df, tools


def extract_graph_df(xml_dir, category, graphs_dir_out, max_size):
    ranking_df, tools = extract_ranking_df(xml_dir, category, max_size)
    verification_tasks = []
    for (index, _) in ranking_df.iterrows():
        verification_tasks.append(index)
    if not os.path.exists(graphs_dir_out):
        os.makedirs(graphs_dir_out)
    graph_df, graphgen_times = graphs.create_graph_df(verification_tasks, graphs_dir_out)
    ret_df = pd.concat([ranking_df, graph_df], axis=1)
    ret_df.dropna(inplace=True)
    return ret_df, tools, graphgen_times


def write_statistics(file, graphgen_times, total_time, all_categories_df, tools):
    ranking_dict = all_categories_df['ranking'].to_dict()
    spearman = SpearmansRankCorrelation(tools)
    distances = []
    for _, ranking_a in ranking_dict.items():
        for _, ranking_b in ranking_dict.items():
            distances.append(spearman.compute(Ranking(ranking_a), Ranking(ranking_b)))

    with open(file, 'w') as f:
        f.write('Mean time for generation of graphs: {} seconds (Std: {}) \n'.format(np.mean(graphgen_times), np.std(graphgen_times)))
        f.write('Median time for generation of graphs: {} seconds \n'.format(np.median(graphgen_times)))
        f.write('Total time: {} seconds \n'.format(total_time))
        f.write('Number of examples: {} \n'.format(len(all_categories_df.index)))
        f.write('Mean pairwise Spearman: {} (Std {}) \n'.format(np.mean(distances), np.std(distances)))
        f.write('Median pairwise Spearman: {}'.format(np.median(distances)))


def main():
    parser = argparse.ArgumentParser(description='Create RPC Input from SVCOMP data')
    parser.add_argument('--graphs', type=str, required=False)
    parser.add_argument('--verifolio', type=str, required=False)
    parser.add_argument('-c', '--categories', nargs='+', type=str, required=False)
    parser.add_argument('-g', '--graphs_dir_out', type=str, required=False)
    parser.add_argument('-o', '--df_out', type=str, required=True)
    parser.add_argument('--max_size', type=int, required=False)
    parser.add_argument('--hist', type=str, required=False)

    args = parser.parse_args()
    if all([args.graphs, args.graphs_dir_out, args.categories, args.max_size]):

        start_time = time.time()

        category_df_list = []
        tools_set = set()
        graphgen_times = []
        for category in args.categories:
            df, tools, gg_times = extract_graph_df(args.graphs, category, args.graphs_dir_out, args.max_size)
            category_df_list.append(df)
            tools_set.update(tools)
            graphgen_times.extend(gg_times)

        all_categories_df = pd.concat(category_df_list)
        all_categories_df.to_csv(args.df_out)
        with open(args.df_out + '.tools', 'w') as f:
            f.write(",".join(tools_set))

        total_time = time.time() - start_time
        write_statistics(args.df_out + '.statistics', graphgen_times, total_time, all_categories_df, tools)

    elif all([args.hist, args.df_out]):
        df = pd.DataFrame.from_csv(args.hist)
        graph_paths = df['graph_representation'].tolist()
        graph_list = []
        for p in graph_paths:
            if not os.path.isfile(p):
                raise ValueError('Graph {} not found.'.format(p))
            g = nx.read_gpickle(p)
            graph_list.append(g)
        graphinfo.generate_node_number_hist(graph_list)
        graphinfo.generate_edge_number_hist(graph_list)
        graphinfo.generate_edge_number_hist(graph_list, 1)
        graphinfo.generate_edge_number_hist(graph_list, 2)
        graphinfo.generate_edge_number_hist(graph_list, 3)
        graphinfo.generate_edge_number_hist(graph_list, 4)
        graphinfo.generate_node_depth_hist(graph_list)

    elif all([args.verifolio, args.categories, args.max_size]):
        category_df_list = []
        tools_set = set()
        for category in args.categories:
            df, tools = extract_feature_df(args.verifolio, category, args.max_size)
            category_df_list.append(df)
            tools_set.update(tools)

        all_categories_df = pd.concat(category_df_list)
        all_categories_df.to_csv(args.df_out)
        with open(args.df_out + '.tools', 'w') as f:
            f.write(",".join(tools_set))
    else:
        parser.print_usage()
        quit()


if __name__ == '__main__':
    main()


import argparse
import os
import pandas as pd
from PyPRSVT.preprocessing import ranking, svcomp15, verifolio, graphs


def c_or_i(f):
    return f.endswith('.i') or f.endswith('.c')


def extract_ranking_df(xml_dir, category):
        results = svcomp15.read_category(xml_dir, category)
        return ranking.create_ranking_df(results, svcomp15.compare_results)


def extract_graph_df(xml_dir, category, graphs_dir_out):
        ranking_df, tools = extract_ranking_df(xml_dir, category)
        verification_tasks = []
        for (index, _) in ranking_df.iterrows():
            verification_tasks.append(index)
        if not os.path.exists(graphs_dir_out):
            os.makedirs(graphs_dir_out)
        graph_df = graphs.create_graph_df(verification_tasks, graphs_dir_out)
        ret_df = pd.concat([ranking_df, graph_df], axis=1)
        ret_df.dropna(inplace=True)
        return ret_df, tools


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create RPC Input from SVCOMP data')
    parser.add_argument('--graphs', type=str, required=False)
    parser.add_argument('-c', '--category', type=str, required=True)
    parser.add_argument('-g', '--graphs_dir_out', type=str, required=False)
    parser.add_argument('-o', '--df_out', type=str, required=True)
    args = parser.parse_args()
    if all([args.graphs, args.graphs_dir_out]):
        df, tools = extract_graph_df(args.graphs, args.category, args.graphs_dir_out)
        df.to_csv(args.df_out)
        with open(args.df_out + '.tools', 'w') as f:
            f.write(",".join(tools))

    else:
        parser.print_usage()
        quit()



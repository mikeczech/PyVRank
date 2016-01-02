import argparse
import os
import pandas as pd
from PyPRSVT.preprocessing import ranking, svcomp15, verifolio, graphs


def c_or_i(f):
    return f.endswith('.i') or f.endswith('.c')


def extract_ranking_df(xml_dir, category, max_size):
    results = svcomp15.read_category(xml_dir, category, max_size)
    return ranking.create_ranking_df(results, svcomp15.compare_results)


def extract_graph_df(xml_dir, category, graphs_dir_out, max_size):
    ranking_df, tools = extract_ranking_df(xml_dir, category, max_size)
    verification_tasks = []
    for (index, _) in ranking_df.iterrows():
        verification_tasks.append(index)
    if not os.path.exists(graphs_dir_out):
        os.makedirs(graphs_dir_out)
    graph_df = graphs.create_graph_df(verification_tasks, graphs_dir_out)
    ret_df = pd.concat([ranking_df, graph_df], axis=1)
    ret_df.dropna(inplace=True)
    return ret_df, tools


def main():
    parser = argparse.ArgumentParser(description='Create RPC Input from SVCOMP data')
    parser.add_argument('--graphs', type=str, required=False)
    parser.add_argument('-c', '--categories', nargs='+', type=str, required=True)
    parser.add_argument('-g', '--graphs_dir_out', type=str, required=False)
    parser.add_argument('-o', '--df_out', type=str, required=True)
    parser.add_argument('--max_size', type=int, required=False)
    args = parser.parse_args()
    if all([args.graphs, args.graphs_dir_out, args.categories, args.max_size]):

        category_df_list = []
        tools_set = set()
        for category in args.categories:
            df, tools = extract_graph_df(args.graphs, category, args.graphs_dir_out, args.max_size)
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


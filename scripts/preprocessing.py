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



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Create RPC Input from SVCOMP data')
#     parser.add_argument('-c', '--category', type=str, nargs='+', required=False)
#     parser.add_argument('-o', '--output', type=str, required=True)
#     parser.add_argument('-d', '--dir', type=str, required=False)
#     parser.add_argument('--cpachecker', type=str, required=False)
#     parser.add_argument('--csv', type=str, required=False)
#     parser.add_argument('--rankings', action='store_true', required=False)
#     parser.add_argument('--features', action='store_true', required=False)
#     parser.add_argument('--cfg', action='store_true', required=False)
#     parser.add_argument('--gram', action='store_true', required=False)
#     args = parser.parse_args()
#     if not any([args.rankings, args.features, args.cfg, args.gram]):
#         parser.print_usage()
#         quit()
#     # Write data set with tool preferences to output
#     if all([args.rankings, args.category, args.dir]):
#         for c in args.category:
#             print('Processing category ' + c)
#             results = svcomp15.read_category(args.dir, c)
#             df, tools = ranking.create_benchmark_ranking_df(results, svcomp15.compare_results)
#             df.to_csv(args.output + "." + c + ".csv")
#             with open(args.output + '.' + c + '.tools', 'w') as f:
#                 f.write(",".join(tools))
#
#     # Write data set with verifolio features to output
#     elif all([args.features, args.csv, args.output]):
#         source_files = []
#         for csv in args.csv:
#             df = pd.DataFrame.from_csv(csv)
#             for (index, _) in df.iterrows():
#                 source_files.append(index)
#         ret = verifolio.create_feature_df(set(source_files))
#         ret.to_csv(args.output)
#
#     elif all([args.cfg, args.csv, args.category, args.dir, args.cpachecker, args.output]):
#         for c in args.category:
#             print('Processing category ' + c)
#             source_files = []
#             df = pd.DataFrame.from_csv(args.csv + '.' + c + '.csv')
#             for (index, _) in df.iterrows():
#                 source_files.append(index)
#             cfg = LabeledDiGraphGen(args.cpachecker)
#             out_dir = join(args.dir, c)
#             if not os.path.exists(out_dir):
#                 os.makedirs(out_dir)
#             ret = cfg.create_digraph_df(source_files, out_dir)
#             ret.to_csv(args.output + '.' + c + '.csv')
#
#     elif all([args.gram, args.csv, args.output]):
#         for csv in args.csv:
#             graphs_df = pd.DataFrame.from_csv(csv)
#             graph_list = []
#             for _, row in graphs_df.iterrows():
#                 print('Processing ' + row.iloc[0])
#                 nx_digraph = nx.read_dot(row.iloc[0])
#                 graph_list.append(nx_digraph)
#             X = gk.compare_list_normalized(graph_list, h=2)
#             np.save(args.output, X)
#
#     # Wrong arguments, therefore print usage
#     else:
#         parser.print_usage()
#         quit()

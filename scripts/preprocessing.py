from PyPRSVT.preprocessing.competition import svcomp15
from PyPRSVT.preprocessing import ranking
from PyPRSVT.preprocessing.verifolio import features as vf
import argparse
import pandas as pd
import os
from os.path import join, isfile


def c_or_i(f):
    return f.endswith('.i') or f.endswith('.c')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create RPC Input from SVCOMP data')
    parser.add_argument('-c', '--category', type=str, nargs='+', required=False)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-d', '--dir', type=str, required=False)
    parser.add_argument('-R', '--recursive', action='store_true', required=False)
    parser.add_argument('--fromcsv', nargs='+', required=False)
    parser.add_argument('--preferences', action='store_true', required=False)
    parser.add_argument('--features', action='store_true', required=False)
    args = parser.parse_args()
    if not any([args.preferences, args.features]):
        parser.print_usage()
        quit()

    # Write data set with tool preferences to output
    if all([args.preferences, args.category, args.dir]):
        dfs = []
        for c in args.category:
            results = svcomp15.read_category(args.dir, c)
            df, tools = ranking.create_benchmark_ranking_df(results, svcomp15.compare_results)
            dfs.append(df)
        ret = pd.concat(dfs)
        ret.to_csv(args.output)
        with open(args.output + '.tools', 'w') as f:
            f.write(",".join(tools))

    # Write data set with verifolio features to output
    elif args.features and any([args.dir, args.fromcvs]):
        sourcefiles = []
        if args.fromcsv:
            for csv in args.fromcsv:
                df = pd.DataFrame.from_csv(csv)
                for (index, _) in df.iterrows():
                    sourcefiles.append(index)
        elif args.recursive:
            for (dirpath, dirnames, filenames) in os.walk(args.dir):
                for f in filenames:
                    if c_or_i(f):
                        sourcefiles.append(join(dirpath, f))
        else:
            sourcefiles.extend([join(args.dir, f) for f in os.listdir(args.dir)
                                if isfile(join(args.dir, f)) and c_or_i(f)])
        ret = vf.create_feature_df(set(sourcefiles))
        ret.to_csv(args.output)

    # Wrong arguments, therefore print usage
    else:
        parser.print_usage()
        quit()

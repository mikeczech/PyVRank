from PyPRSVT.preprocessing.competition import svcomp15
from PyPRSVT.preprocessing import ranking
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create RPC Input from SVCOMP data')
    parser.add_argument('-r', '--results', type=str, required=True)
    parser.add_argument('-c', '--category', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    results = svcomp15.read_category(args.results, args.category)
    df = ranking.create_benchmark_ranking_df(results, svcomp15.compare_results)
    df.to_csv(args.output)

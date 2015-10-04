import subprocess
from os.path import isfile, join, abspath, isdir, basename
import re
import networkx as nx
import pandas as pd


class LabeledDiGraphGen(object):

    def __init__(self, path_to_cpachecker):
        self.path_to_cpachecker = path_to_cpachecker

    def _run_cpachecker(self, path_to_source):
        """
        Runs a CFALabels-Analysis
        :param path_to_cpachecker:
        :param path_to_source:
        :return:
        """
        cpash_path = join(self.path_to_cpachecker, 'scripts', 'cpa.sh')
        output_path = join(self.path_to_cpachecker, 'output')
        cfa_path = join(self.path_to_cpachecker, 'output', 'cfa.dot')
        reached_path = join(self.path_to_cpachecker, 'output', 'reached.txt')
        if not isdir(self.path_to_cpachecker):
            raise ValueError('CPAChecker directory not found')
        if not (isfile(path_to_source) and path_to_source.endswith('.i')):
            raise ValueError('path_to_source is no valid filepath')
        subprocess.call([cpash_path, '-cfalabelsAnalysis', path_to_source, '-outputpath', output_path])
        assert isfile(cfa_path) and isfile(reached_path), 'Invalid output of CPAChecker'
        return cfa_path, reached_path

    @staticmethod
    def _read_relabeling(reached_path):
        """
        Computes relabeling from reached states of CFALabels-Analysis
        :param reached_path:
        :return:
        """
        labels = {}
        with open(reached_path) as f:
            for line in f:
                m = re.match(r"\s*CFALabelsState:\s\[([0-9]+),\s([0-9]+),\s\[([A-Z_,\s]+)\]\]\n", line)
                if m is not None:
                    edge = (m.group(1), m.group(2), 0)
                    label_list = sorted([x.strip() for x in m.group(3).split(',')])
                    labels[edge] = "_".join(label_list)
        return labels

    def create_digraph_df(self, source_path_list, dest_dir_path):
        """
        Creates a frame that maps sourcefiles to networkx digraphs in terms of DOT files
        :param source_path_list:
        :param dest_dir_path:
        :param relabel:
        :return:
        """
        if not isdir(dest_dir_path):
            raise ValueError('Invalid destination directory.')
        data = []
        for source_path in source_path_list:
            dot_path = join(dest_dir_path, basename(source_path) + '.dot')
            # DEBUG
            if isfile(dot_path):
                data.append(dot_path)
                continue
            cfa_path, reached_path = self._run_cpachecker(abspath(source_path))
            nx_digraph = nx.read_dot(cfa_path)
            relabeling = LabeledDiGraphGen._read_relabeling(reached_path)
            # Relabel graph
            nx.set_edge_attributes(nx_digraph, 'label', relabeling)
            nx.write_dot(nx_digraph, dot_path)
            data.append(dot_path)
        return pd.DataFrame(data, index=source_path_list)

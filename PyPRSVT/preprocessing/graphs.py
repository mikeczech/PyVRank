import subprocess
from os.path import isfile, join, abspath, isdir, basename
import re
import networkx as nx
import pandas as pd

path_to_cpachecker = ""


def _run_cpachecker(self, path_to_source):
    """
    Runs a CFALabels-Analysis
    :param path_to_cpachecker:
    :param path_to_source:
    :return:
    """
    cpash_path = join(self.path_to_cpachecker, 'scripts', 'cpa.sh')
    output_path = join(self.path_to_cpachecker, 'output')
    graph_path = join(self.path_to_cpachecker, 'output', 'vtask_graph.dot')
    if not isdir(self.path_to_cpachecker):
        raise ValueError('CPAChecker directory not found')
    if not (isfile(path_to_source) and (path_to_source.endswith('.i') or path_to_source.endswith('.c'))):
        raise ValueError('path_to_source is no valid filepath')
    out = subprocess.check_output([cpash_path, '-cfalabelsAnalysis', path_to_source, '-outputpath', output_path])
    match_vresult = re.search(r'Verification\sresult:\s([A-Z]+)\.', str(out))
    if match_vresult is None:
        raise ValueError('Invalid output of CPAChecker.')
    if match_vresult.group(1) != 'TRUE':
        raise ValueError('CFALabels Analysis failed:' + out)
    assert isfile(graph_path), 'Invalid output of CPAChecker'
    return graph_path


def create_graph_df(vtask_paths, graphs_dir_out):
    """
    Creates a frame that maps sourcefiles to networkx digraphs in terms of DOT files
    :param source_path_list:
    :param dest_dir_path:
    :param relabel:
    :return:
    """
    if not isdir(graphs_dir_out):
        raise ValueError('Invalid destination directory.')
    data = []
    for vtask in vtask_paths:
        ret_path = join(graphs_dir_out, basename(vtask) + '.dot')
        # DEBUG
        if isfile(ret_path):
            data.append(ret_path)
            continue
        graph_path = _run_cpachecker(abspath(vtask))
        nx_digraph = nx.read_dot(graph_path)
        assert not isfile(ret_path)
        nx.write_dot(nx_digraph, ret_path)
        data.append(ret_path)
    return pd.DataFrame(data, index=vtask_paths)

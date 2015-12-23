import subprocess
from os.path import isfile, join, abspath, isdir, basename
import re
import networkx as nx
import pandas as pd
from enum import Enum


class EdgeType(Enum):
    de = 1
    ce = 2
    cfe = 3
    se = 4

__path_to_cpachecker__ = ""


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
    node_labels_path = join(self.path_to_cpachecker, 'output', 'nodes.labels')
    edge_types_path = join(self.path_to_cpachecker, 'output', 'edge_types.labels')
    edge_truth_path = join(self.path_to_cpachecker, 'output', 'edge_truth.labels')
    node_depths_path = join(self.path_to_cpachecker, 'output', 'edge_depths.labels')
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
    if not isfile(graph_path):
        raise ValueError('Invalid output of CPAChecker: Missing graph output')
    if not isfile(node_labels_path):
        raise ValueError('Invalid output of CPAChecker: Missing node labels output')
    if not isfile(edge_types_path):
        raise ValueError('Invalid output of CPAChecker: Missing edge types output')
    if not isfile(edge_truth_path):
        raise ValueError('Invalid output of CPAChecker: Missing edge truth values output')
    if not isfile(node_depths_path):
        raise ValueError('Invalid output of CPAChecker: Missing node depths output')
    return graph_path, node_labels_path, edge_types_path, edge_truth_path, node_depths_path


def _read_node_labeling(labels_path):
    labels = {}
    with open(labels_path) as f:
        for line in f:
            m = re.match(r"([0-9]+),([A-Z_]+)\n", line)
            if m is not None:
                node = m.group(1)
                labels[node] = m.group(2)
    return labels


def _read_edge_labeling(labels_path):
    labels = {}
    with open(labels_path) as f:
        for line in f:
            m = re.match(r"([0-9]+),([0-9]+),\[(([A-Z_\s]+,?)+)\]\n", line)
            if m is not None:
                edge = (m.group(1), m.group(2), 0)
                labels[edge] = [x.strip() for x in m.group(2).split(',')]
    return labels


def _parse_edge_types(edge_types):
    types = {}
    str_to_type_map = {'DE': EdgeType.de, 'CE': EdgeType.ce, 'SE': EdgeType.se, 'CFE': EdgeType.cfe}
    for edge, l in edge_types.iteritems():
        if l not in str_to_type_map:
            raise ValueError('Unknown edge type ' + l + '. Wrong input?')
        types[edge] = str_to_type_map[l]
    return types



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
        ret_path = join(graphs_dir_out, basename(vtask) + '.graphml')

        # DEBUG
        if isfile(ret_path):
            data.append(ret_path)
            continue

        graph_path, node_labels_path, edge_types_path, edge_truth_path, node_depths_path \
            = _run_cpachecker(abspath(vtask))
        nx_digraph = nx.read_dot(graph_path)
        node_labels = _read_node_labeling(node_labels_path)
        nx.set_node_attributes(nx_digraph, 'label', node_labels)
        edge_types = _read_edge_labeling(edge_types_path)
        parsed_edge_types = _parse_edge_types(edge_types)
        nx.set_edge_attributes(nx_digraph, 'types', parsed_edge_types)
        edge_truth = _read_edge_labeling(edge_truth_path)
        nx.set_edge_attributes(nx_digraph, 'truth', edge_truth)
        node_depths = _read_node_labeling(node_depths_path)
        nx.set_node_attributes(nx_digraph, 'depth', node_depths)

        assert not isfile(ret_path)
        nx.write_graphml(nx_digraph, ret_path)
        data.append(ret_path)

    return pd.DataFrame(data, index=vtask_paths)

import subprocess
from subprocess import PIPE
from os.path import isfile, join, abspath, isdir, commonprefix, dirname
import os
import re
import networkx as nx
import pandas as pd
from enum import Enum
from tqdm import tqdm


class EdgeType(Enum):
    de = 1
    ce = 2
    cfe = 3
    se = 4
    t = 5
    f = 6
    dummy = 7

__path_to_cpachecker__ = "/home/mike/Documents/Repositories/cpachecker"


def _run_cpachecker(path_to_source):
    """
    Runs a CFALabels-Analysis
    :param path_to_cpachecker:
    :param path_to_source:
    :return:
    """
    cpash_path = join(__path_to_cpachecker__, 'scripts', 'cpa.sh')
    output_path = join(__path_to_cpachecker__, 'output')
    graph_path = join(__path_to_cpachecker__, 'output', 'vtask_graph.graphml')
    node_labels_path = join(__path_to_cpachecker__, 'output', 'nodes.labels')
    edge_types_path = join(__path_to_cpachecker__, 'output', 'edge_types.labels')
    edge_truth_path = join(__path_to_cpachecker__, 'output', 'edge_truth.labels')
    node_depths_path = join(__path_to_cpachecker__, 'output', 'node_depth.labels')
    if not isdir(__path_to_cpachecker__):
        raise ValueError('CPAChecker directory not found')
    if not (isfile(path_to_source) and (path_to_source.endswith('.i') or path_to_source.endswith('.c'))):
        raise ValueError('path_to_source is no valid filepath')
    try:
        proc = subprocess.run([cpash_path, '-graphgenAnalysis', '-skipRecursion', path_to_source, '-outputpath', output_path],
                              check=False, stdout=PIPE, stderr=PIPE)
        match_vresult = re.search(r'Verification\sresult:\s([A-Z]+)\.', str(proc.stdout))
        if match_vresult is None:
            raise ValueError('Invalid output of CPAChecker.')
        if match_vresult.group(1) != 'TRUE':
            raise ValueError('ASTCollector Analysis failed:')
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
    except ValueError as err:
        print(err)
        print(proc.args)
        print(proc.stdout.decode('utf-8'))
        print(proc.stderr.decode('utf-8'))
        quit()

    return graph_path, node_labels_path, edge_types_path, edge_truth_path, node_depths_path


def _read_node_labeling(labels_path):
    labels = {}
    with open(labels_path) as f:
        for line in f:
            m = re.match(r"([0-9]+),([A-Z_0-9]+)\n", line)
            if m is not None:
                node = m.group(1)
                labels[node] = m.group(2)
    return labels


def _read_edge_labeling(labels_path):
    labels = {}
    with open(labels_path) as f:
        for line in f:
            m = re.match(r"([0-9]+),([0-9]+),([0-9]+),([0-9]+)\n", line)
            if m is not None:
                # Todo add for multigraphs
                edge = (m.group(1), m.group(2), m.group(3))
                # edge = (m.group(1), m.group(2))
                labels[edge] = m.group(4)
    return labels


def _parse_edge(edge_types):
    types = {}
    for edge, l in edge_types.items():
        types[edge] = EdgeType(int(l))
    return types


def _parse_node_depth(node_depth):
    types = {}
    for node, l in node_depth.items():
        types[node] = int(l)
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

    print('Writing graph representations of verification tasks to {}'.format(graphs_dir_out), flush=True)

    common_prefix = commonprefix(vtask_paths)
    for vtask in tqdm(vtask_paths):
        short_prefix = dirname(common_prefix)
        path = join(graphs_dir_out, vtask[len(short_prefix):][1:])

        if not os.path.exists(dirname(path)):
            os.makedirs(dirname(path))

        ret_path = path + '.pickle'

        # DEBUG
        if isfile(ret_path):
            data.append(ret_path)
            continue

        graph_path, node_labels_path, edge_types_path, edge_truth_path, node_depths_path \
            = _run_cpachecker(abspath(vtask))
        nx_digraph = nx.read_graphml(graph_path)

        node_labels = _read_node_labeling(node_labels_path)
        nx.set_node_attributes(nx_digraph, 'label', node_labels)

        edge_types = _read_edge_labeling(edge_types_path)
        parsed_edge_types = _parse_edge(edge_types)
        nx.set_edge_attributes(nx_digraph, 'type', parsed_edge_types)

        edge_truth = _read_edge_labeling(edge_truth_path)
        parsed_edge_truth = _parse_edge(edge_truth)
        nx.set_edge_attributes(nx_digraph, 'truth', parsed_edge_truth)

        node_depths = _read_node_labeling(node_depths_path)
        parsed_node_depths = _parse_node_depth(node_depths)
        nx.set_node_attributes(nx_digraph, 'depth', parsed_node_depths)

        assert not isfile(ret_path)
        assert node_labels and parsed_edge_types and parsed_edge_truth and parsed_node_depths
        nx.write_gpickle(nx_digraph, ret_path)
        data.append(ret_path)

    return pd.DataFrame({'graph_representation': data}, index=vtask_paths)

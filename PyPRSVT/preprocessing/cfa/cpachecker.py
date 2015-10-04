import subprocess
from os.path import isfile, join
import re


def run_cpachecker(path_to_cpachecker, path_to_source):
    """
    Runs a CFALabels-Analysis
    :param path_to_cpachecker:
    :param path_to_source:
    :return:
    """
    cpash_path = join(path_to_cpachecker, 'scripts', 'cpa.sh')
    cfa_path = join(path_to_cpachecker, 'output', 'cfa.dot')
    reached_path = join(path_to_cpachecker, 'output', 'reached.txt')
    if not isfile(path_to_cpachecker):
        raise ValueError('CPAChecker not found')
    if not (isfile(path_to_source) and path_to_source.endswith('.i')):
        raise ValueError('path_to_source is no valid filepath')
    subprocess.call([cpash_path, '-cfalabelsAnalysis', path_to_source])
    assert isfile(cfa_path) and isfile(reached_path), 'Invalid output of CPAChecker'
    return cfa_path, reached_path


def read_relabeling(reached_path):
    """
    Computes relabeling from reached states of CFALabels-Analysis
    :param reached_path:
    :return:
    """
    labels = {}
    with open(reached_path) as f:
        for line in f:
            m = re.match(r"\s*CFALabelsState:\s\[([0-9]+),\s([0-9]+)\s\[([A-Z,\s]+)\]\]", line)
            if m is not None:
                edge = (int(m.group(1)), int(m.group(2)))
                label_list = sorted([x.strip() for x in m.group(3).split(',')])
                labels[edge] = "_".join(label_list)
    return labels


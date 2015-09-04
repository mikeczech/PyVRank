from lxml import objectify
import re
import os
import pandas as pd

import PyPRSVT.preprocessing.utils as utils
import PyPRSVT.basics as b


class MissingPropertyTypeException(Exception):
    pass

class MissingExpectedStatusException(Exception):
    pass


def read_category(results_xml_raw_dir_path, category):
    """
    Reads a directory of raw xml SVCOMP results into a data frame.

    :param results_xml_raw_dir_path: Path to the raw xml results from SVCOMP
    :return: Pandas data frame
    """
    pattern = re.compile(r'\w+\.[0-9-_]+\.results\.sv-comp15\.{0}\.xml'.format(category))
    category_results = []
    for file in os.listdir(results_xml_raw_dir_path):
        if re.match(pattern, file) is not None:
            category_results.append(read_results(os.path.join(results_xml_raw_dir_path, file)))
    return pd.concat(dict(category_results), axis=1)


def read_results(results_xml_raw_path):
    """
    Reads raw xml SVCOMP results into a data frame.

    :param results_xml_raw_path:
    :return:
    """
    with open(results_xml_raw_path) as f:
        xml = f.read()
    root = objectify.fromstring(xml)
    df = pd.DataFrame(columns=['options', 'status', 'status_msg', 'cputime', 'walltime', 'mem_usage',
                               'expected_status', 'property_type'])
    for source_file in root.sourcefile:
        vtask_path = source_file.attrib['name']
        r = columns_to_dict(source_file.column)
        df.loc[vtask_path] = [source_file.attrib['options'] if 'options' in source_file.attrib else '',
                              utils.match_status_str(r['status']),
                              r['status'],
                              float(r['cputime'][:-1]),
                              float(r['walltime'][:-1]),
                              int(r['memUsage']),
                              extract_expected_status(vtask_path),
                              extract_property_type(vtask_path)]
    return root.attrib['tool'], df


def columns_to_dict(columns):
    """
    Simple helper function, which converts column tags to a dictionary.
    :param columns: Collection of column tags
    :return: Dictionary that contains all the information from the columns.
    """
    ret = {}
    for column in columns:
        ret[column.attrib['title']] = column.attrib['value']
    assert ret, 'Could not read columns from sourcefile'
    return ret


def extract_expected_status(vtask_path):
    """
    Extracts the expected status from a verification task.

    :param vtask_path: Path to a SVCOMP verification task.
    :return: A tuple containing a verification task's expected result and
             property type if the filename adheres the naming convention.
             TODO What is the exact naming convention?

             Otherwise the result is None.
    """
    match = re.match(r'[-a-zA-Z0-9_\.]+_(true|false)-([-a-zA-Z0-9_]+)\.(i|c)',
                     os.path.basename(vtask_path))
    if match is not None:
        return utils.match_status_str(match.group(1))
    raise MissingExpectedStatusException('Cannot extract expected status from filename / regex failed (wrong naming?)')


def extract_property_type(vtask_path):
    """
    Extracts the property type associated with a verification task.
    :param vtask_path: path to verification task
    :return: the property type
    """
    unreachability_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*G\s*!\s*call\([_\w\s\(\)]+\)\s*\)\s*\)')
    memory_safety_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*G\s*valid-\w+\)\s*\)')
    termination_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*F\s*end\s*\)\s*\)')

    root, ext = os.path.splitext(vtask_path)
    prp = root + '.prp'
    if not os.path.isfile(prp):
        prp = os.path.join(os.path.dirname(vtask_path), 'ALL.prp')
    if not os.path.isfile(prp):
        raise MissingPropertyTypeException('Missing ALL.prp or {filename}.prp')

    with open(prp) as f:
        prp_file_content = f.read()
    if unreachability_pattern.search(prp_file_content) is not None:
        return b.PropertyType.unreachability
    if memory_safety_pattern.search(prp_file_content) is not None:
        return b.PropertyType.memory_safety
    if termination_pattern.search(prp_file_content) is not None:
        return b.PropertyType.termination
    raise MissingPropertyTypeException('Cannot determine property type from prp file')

from lxml import objectify
from collections import namedtuple
from enum import Enum, unique
import re
import os.path

Result = namedtuple('Result', 'sourcefile status status_msg cputime walltime mem_usage expected_status property_type')


@unique
class PropertyType(Enum):
    unreachability = 1
    memory_safety = 2
    termination = 3
    undefined = 4


@unique
class Status(Enum):
    true = 1
    false = 2
    unknown = 3


def read_results_dir(results_xml_raw_dir_path):
    """
    Reads a directory of raw xml SVCOMP results into a data frame.

    :param results_xml_raw_dir_path: Path to the raw xml results from SVCOMP
    :return: Pandas data frame
    """
    pass


def read_results(results_xml_raw_path):
    """
    Reads raw xml SVCOMP results into a data frame.

    :param results_xml_raw_path:
    :return:
    """
    with open(results_xml_raw_path) as f:
        xml = f.read()

    root = objectify.fromstring(xml)
    for source_file in root.sourcefile:
        r = columns_to_dict(source_file.column)
        vtask_path = source_file.attrib['name']
        yield Result(vtask_path,
                     match_status_str(r['status']),
                     r['status'],
                     r['cputime'],
                     r['walltime'],
                     r['memUsage'],
                     extract_expected_status(vtask_path),
                     extract_property_type(vtask_path))


def columns_to_dict(columns):
    """
    Simple helper function, which converts column tags to a dictionary.
    :param columns: Collection of column tags
    :return: Dictionary that contains all the information from the columns.
    """
    ret = {}
    for column in columns:
        ret[column.attrib['title']] = column.attrib['value']
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
    match = re.match(r'[-a-zA-Z0-9_]+_(true|false)-([-a-zA-Z0-9_]+)\.(i|c)',
                     os.path.basename(vtask_path))
    if match is not None:
        return match_status_str(match.group(1))
    return None


def match_status_str(status_str):
    """
    Maps status strings to their associated meaning
    :param status_str: the status string
    :return: true, false, or unknown
    """
    if status_str == 'true':
        return Status.true
    if status_str == 'false':
        return Status.false
    else:
        return Status.unknown


def extract_property_type(vtask_path):
    """
    Extracts the property type associated from a verification task.
    :param vtask_path: path to verification task
    :return: the property type
    """
    unreachability_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*G\s*!\s*call\([_\w\s\(\)]+\)\)')
    memory_safety_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*G\s*valid-\w+\)\)')
    termination_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*F\s*end\)\)')

    root, ext = os.path.splitext(vtask_path)
    prp = root + '.prp'
    if not os.path.isfile(prp):
        prp = os.path.join(os.path.dirname(vtask_path), 'ALL.prp')
    if not os.path.isfile(prp):
        return PropertyType.undefined

    with open(prp) as f:
        prp_file_content = f.read()
    if unreachability_pattern.search(prp_file_content) is not None:
        return PropertyType.unreachability
    if memory_safety_pattern.search(prp_file_content) is not None:
        return PropertyType.memory_safety
    if termination_pattern.search(prp_file_content) is not None:
        return PropertyType.termination
    return PropertyType.undefined

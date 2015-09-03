from lxml import objectify
from collections import namedtuple
import re
import ntpath

Result = namedtuple('Result', 'sourcefile status cputime walltime mem_usage expected_result property_type')

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
        name = source_file.attrib['name']
        expected_result, property_type = extract_attributes(name)
        yield Result(name,
                     r['status'],
                     r['cputime'],
                     r['walltime'],
                     r['memUsage'],
                     expected_result,
                     property_type)


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


def extract_attributes(vtask_path):
    """
    Extracts the expected result and property type from a verification task's
    filename.

    :param vtask_path: Path to a SVCOMP verification task.
    :return: A tuple containing a verification task's expected result and
             property type if the filename adheres the naming convention.
             TODO What is the exact naming convention?

             Otherwise the result is None.
    """
    x = 1
    match = re.match(r'[-a-zA-Z0-9_]+_(true|false)-([-a-zA-Z0-9_]+)\.(i|c)',
                     ntpath.basename(vtask_path))
    if match is not None:
        return match.group(1), match.group(2)
    return None


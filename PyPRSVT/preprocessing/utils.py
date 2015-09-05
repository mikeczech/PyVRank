import PyPRSVT.basics as b
import re

def match_status_str(status_str):
    """
    Maps status strings to their associated meaning
    :param status_str: the status string
    :return: true, false, or unknown
    """
    if re.search(r'true', status_str):
        return b.Status.true
    if re.search(r'false', status_str):
        return b.Status.false
    else:
        return b.Status.unknown

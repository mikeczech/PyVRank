import PyPRSVT.basics as b

def match_status_str(status_str):
    """
    Maps status strings to their associated meaning
    :param status_str: the status string
    :return: true, false, or unknown
    """
    if status_str == 'true':
        return b.Status.true
    if status_str == 'false':
        return b.Status.false
    else:
        return b.Status.unknown

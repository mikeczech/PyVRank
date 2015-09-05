import nose.tools as nt
import PyPRSVT.preprocessing.utils as utils
import PyPRSVT.basics as b


def match_status_str_test():
    nt.assert_equal(utils.match_status_str('true'), b.Status.true)
    nt.assert_equal(utils.match_status_str('false'), b.Status.false)
    nt.assert_equal(utils.match_status_str('unknown'), b.Status.unknown)
    nt.assert_equal(utils.match_status_str('TIMEOUT'), b.Status.unknown)
    nt.assert_equal(utils.match_status_str('OUT OF MEMORY'), b.Status.unknown)
    nt.assert_equal(utils.match_status_str('false(reach)'), b.Status.false)
    nt.assert_equal(utils.match_status_str('error'), b.Status.unknown)
    nt.assert_equal(utils.match_status_str('EXCEPTION (Gremlins)'), b.Status.unknown)


import nose.tools as nt
import PyPRSVT.preprocessing.svcomp as svcomp


def match_status_str_test():
    nt.assert_equal(svcomp.match_status_str('true'), svcomp.Status.true)
    nt.assert_equal(svcomp.match_status_str('false'), svcomp.Status.false)
    nt.assert_equal(svcomp.match_status_str('unknown'), svcomp.Status.unknown)
    nt.assert_equal(svcomp.match_status_str('TIMEOUT'), svcomp.Status.unknown)
    nt.assert_equal(svcomp.match_status_str('OUT OF MEMORY'), svcomp.Status.unknown)
    nt.assert_equal(svcomp.match_status_str('false(reach)'), svcomp.Status.false)
    nt.assert_equal(svcomp.match_status_str('error'), svcomp.Status.unknown)
    nt.assert_equal(svcomp.match_status_str('EXCEPTION (Gremlins)'), svcomp.Status.unknown)


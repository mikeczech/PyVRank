from PyPRSVT.preprocessing import svcomp
import nose.tools as nt

def read_results_test():
    results = list(svcomp.read_results('static/results-xml-raw/cbmc.14-12-04_1241.results.sv-comp15.Arrays.xml'))

    nt.assert_equal(results[0][0], 'CBMC')
    nt.assert_equal(results[0][1].options, '--32')
    nt.assert_equal(results[0][1].status, svcomp.Status.true)
    nt.assert_equal(results[0][1].cputime, 7.627101835)
    nt.assert_equal(results[0][1].walltime, 7.66226601601)
    nt.assert_equal(results[0][1].mem_usage, 2848837632)
    nt.assert_equal(results[0][1].expected_status, svcomp.Status.false)
    nt.assert_equal(results[0][1].property_type, svcomp.PropertyType.unreachability)

    nt.assert_equal(results[1][1].status_msg, 'OUT OF MEMORY')
    nt.assert_equal(results[1][1].mem_usage, 15000002560)

    nt.assert_equal(results[2][1].walltime, 850.019608021)
    nt.assert_equal(results[2][1].property_type, svcomp.PropertyType.memory_safety)

    nt.assert_equal(results[3][1].property_type, svcomp.PropertyType.termination)
    nt.assert_equal(results[3][1].options, '--64')

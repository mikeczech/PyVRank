from PyPRSVT.preprocessing import svcomp
import nose.tools as nt

def read_results_test():
    benchmark, df = svcomp.read_results('static/results-xml-raw/cbmc.14-12-04_1241.results.sv-comp15.mixed-examples.xml')

    nt.assert_equal(benchmark.tool, 'CBMC')
    nt.assert_equal(benchmark.category, 'sv-comp15.mixed')

    nt.assert_equal(df.iloc[0]['options'], '--32')
    nt.assert_equal(df.iloc[0]['status'], svcomp.Status.true)
    nt.assert_equal(df.iloc[0]['cputime'], 7.627101835)
    nt.assert_equal(df.iloc[0]['walltime'], 7.66226601601)
    nt.assert_equal(df.iloc[0]['mem_usage'], 2848837632)
    nt.assert_equal(df.iloc[0]['expected_status'], svcomp.Status.false)
    nt.assert_equal(df.iloc[0]['property_type'], svcomp.PropertyType.unreachability)

    nt.assert_equal(df.iloc[1]['status_msg'], 'OUT OF MEMORY')
    nt.assert_equal(df.iloc[1]['mem_usage'], 15000002560)

    nt.assert_equal(df.iloc[2]['walltime'], 850.019608021)
    nt.assert_equal(df.iloc[2]['property_type'], svcomp.PropertyType.memory_safety)

    nt.assert_equal(df.iloc[3]['property_type'], svcomp.PropertyType.termination)
    nt.assert_equal(df.iloc[3]['options'], '--64')



# def read_results_dir_test():
#     assert False
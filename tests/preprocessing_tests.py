from PyPRSVT.preprocessing import svcomp
import nose.tools as nt

def read_results_values_test():
    tool, df = svcomp.svcomp_xml_to_dataframe('static/results-xml-raw/cbmc.14-12-04_1241.results.sv-comp15.mixed-examples.xml')

    nt.assert_equal(tool, 'cbmc')

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


def match_status_str_test():
    nt.assert_equal(svcomp._match_status_str('true'), svcomp.Status.true)
    nt.assert_equal(svcomp._match_status_str('false'), svcomp.Status.false)
    nt.assert_equal(svcomp._match_status_str('unknown'), svcomp.Status.unknown)
    nt.assert_equal(svcomp._match_status_str('TIMEOUT'), svcomp.Status.unknown)
    nt.assert_equal(svcomp._match_status_str('OUT OF MEMORY'), svcomp.Status.unknown)
    nt.assert_equal(svcomp._match_status_str('false(reach)'), svcomp.Status.false)
    nt.assert_equal(svcomp._match_status_str('error'), svcomp.Status.unknown)
    nt.assert_equal(svcomp._match_status_str('EXCEPTION (Gremlins)'), svcomp.Status.unknown)


def read_results_no_none_test():
    _, df = svcomp.svcomp_xml_to_dataframe('static/results-xml-raw/cbmc.14-12-04_1241.results.sv-comp15.mixed-examples.xml')
    for __, series in df.iterrows():
        for ___, value in series.iteritems():
            nt.assert_not_equal(value, None)


def read_category_test():
    df = svcomp.read_category('static/results-xml-raw', 'mixed-examples')
    print(df.to_string())

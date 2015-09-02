from PyPRSVT.preprocessing import svcomp

def read_results_test():
    results = svcomp.read_results('static/results-xml-raw/cbmc.14-12-04_1241.results.sv-comp15.Arrays.xml')
    for r in results:
        print(r)


import networkx as nx
from PyPRSVT.gk import GK_WL

def compare_digraphs_test():
    g1 = nx.DiGraph()
    g1.add_edge(1, 2, label='A')
    g1.add_edge(2, 3, label='A')
    g1.add_edge(2, 4, label='B')
    g1.add_edge(4, 5, label='A')
    g1.add_edge(5, 4, label='C')

    g2 = nx.DiGraph()
    g2.add_edge(1, 2, label='B')
    g2.add_edge(2, 3, label='E')
    g2.add_edge(2, 4, label='G')

    print(GK_WL.compare_list([g1, g2], 1))
    print(GK_WL.compare_list_normalized([g1, g2], 1))






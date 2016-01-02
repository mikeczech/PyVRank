import matplotlib.pyplot as plt
import networkx as nx
from PyPRSVT.preprocessing.graphs import EdgeType


def generate_node_number_hist(graphs):
    node_numbers = [g.number_of_nodes() for g in graphs]
    # the histogram of the data
    plt.hist(node_numbers, facecolor='green', bins=100)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')
    plt.show()


def generate_edge_number_hist(graphs, edge_type_num=None):
    edge_numbers = []
    for g in graphs:
        edge_types = nx.get_edge_attributes(g, 'type')
        if edge_type_num is None:
            edge_numbers.append(g.number_of_edges())
        else:
            edge_numbers.append(sum([1 for e in g.edges_iter(keys=True) if edge_types[e] == EdgeType(edge_type_num)]))
    plt.hist(edge_numbers, facecolor='green', bins=300)
    if edge_type_num is None:
        plt.xlabel('Number of Edges')
    else:
        plt.xlabel('Number of Edges of Type ' + str(EdgeType(edge_type_num)))
    plt.ylabel('Frequency')
    plt.show()


def generate_node_depth_hist(graphs):
    depths_list = []
    for g in graphs:
        node_depths = nx.get_node_attributes(g, 'depth')
        depths_list.extend([node_depths[n] for n in g.nodes_iter()])
    # the histogram of the data
    plt.hist(depths_list, facecolor='green', bins=8)
    plt.xlabel('Depth of Nodes')
    plt.ylabel('Frequency')
    plt.show()

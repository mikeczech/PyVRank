import numpy as np
import networkx as nx

class GK_WL(object):
    """
    Weisfeiler Lehman graph kernel for CFGs
    """
    @staticmethod
    def __edge_neighbors(edge, graph):
        src_node, _ = edge
        return graph.in_edges(nbunch=src_node)

    @staticmethod
    def __graph_to_dot(graph, edge_graph_labels, dot_file):
        g_copy = graph.copy()
        nx.set_edge_attributes(g_copy, 'label', edge_graph_labels)
        nx.write_dot(g_copy, dot_file)


    def compare_list_normalized(self, graph_list, h):
        """
        Normalized kernel
        :param graph_list:
        :param h:
        :return:
        """
        k = self.compare_list(graph_list, h)
        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        return k_norm

    def compare_list(self, graph_list, h=1):
        """
        Compute the all-pairs kernel values for a list of labeled CFGs
        :param graph_list:
        :param h:
        :return: similarity matrix of all the graphs in graph_list
        """
        k = [0] * (h+1)
        all_graphs_number_of_edges = 0
        all_graphs_max_number_of_edges = 0
        edge_neighbors = [0] * len(graph_list)
        edge_labels = [0] * (h+1)

        for it in range(h+1):
            edge_labels[it] = [0] * len(graph_list)

        for i, g in enumerate(graph_list):
            edge_neighbors[i] = {e: GK_WL.__edge_neighbors(e, g) for e in g.edges_iter()}
            edge_labels[0][i] = nx.get_edge_attributes(g, 'label')
            all_graphs_number_of_edges += nx.number_of_edges(g)
            if nx.number_of_edges(g) > all_graphs_max_number_of_edges:
                all_graphs_max_number_of_edges = nx.number_of_edges(g)
            GK_WL.__graph_to_dot(g, edge_labels[0][i], "graph{}.dot".format(i))
        phi = np.zeros((all_graphs_number_of_edges, len(graph_list)), dtype=np.uint64)

        label_lookup = {}
        label_counter = 0

        for i, g in enumerate(graph_list):
            for e in g.edges_iter():
                l = edge_labels[0][i][e]
                if l not in label_lookup:
                    label_lookup[l] = label_counter
                    label_counter += 1
                phi[label_lookup[l], i] += 1

        k += np.dot(phi.transpose(), phi)
        print(k)

        for it in range(1, h+1):
            # Todo check if the shape fits in all cases
            phi = np.zeros((2*all_graphs_number_of_edges, len(graph_list)))

            for i, g in enumerate(graph_list):
                edge_labels[it][i] = nx.get_edge_attributes(g, 'label')
                for e in g.edges_iter():
                    long_label = "_".join(np.concatenate([np.sort([edge_labels[it-1][i][nb] for nb in edge_neighbors[i][e]]),
                                                         np.array([edge_labels[it-1][i][e]])]))
                    if long_label not in label_lookup:
                        label_lookup[long_label] = label_counter
                        label_counter += 1
                    edge_labels[it][i][e] = long_label
                    phi[label_lookup[long_label], i] += 1
                GK_WL.__graph_to_dot(g, edge_labels[it][i], "graph{}_it{}.dot".format(i, it))

            print(phi)
            print(label_lookup)
            print(np.dot(phi.transpose(), phi))

            k += np.dot(phi.transpose(), phi)

        return k




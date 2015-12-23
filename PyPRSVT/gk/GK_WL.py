"""
Weisfeiler Lehman graph kernel for CFGs
"""
import numpy as np
import networkx as nx


class GK_WL(object):

    def __init__(self):
        self.__label_lookup__ = {}
        self.__label_counter__ = 0

    def _compress(self, long_label):
        if long_label not in self.__label_lookup__:
            self.__label_lookup__[long_label] = self.__label_counter__
            self.__label_counter__ += 1
        return self.__label_lookup__[long_label]

    def _collect_labels(self, node, i, graph, it, node_labels, node_depth, types, D):
        ret = []
        edge_types = nx.get_edge_attributes(graph, 'types')
        edge_truth = nx.get_edge_attributes(graph, 'truth')
        for e in graph.in_edges(nbunch=node):
            source, _, _ = e
            for t in edge_types[e].split(','):
                if t in types and node_depth[source] <= D:
                    long_edge_label = "".join([node_labels[it][i][source], t, edge_truth[e]])
                    ret.append(self._compress(long_edge_label))
        return ret

    @staticmethod
    def _graph_to_dot(graph, edge_graph_labels, dot_file):
        g_copy = graph.copy()
        nx.set_edge_attributes(g_copy, 'label', edge_graph_labels)
        nx.write_dot(g_copy, dot_file)

    def compare_list_normalized(self, graph_list, types, h, D):
        """
        Normalized kernel
        """
        k = self.compare_list(graph_list, types, h, D)
        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        return k_norm

    def compare_list(self, graph_list, types, h, D):
        """
        Compute the all-pairs kernel values for a list of graph representations of verification tasks
        """
        all_graphs_number_of_nodes = 0
        node_labels = [0] * (h+1)
        node_depth = [0] * len(graph_list)

        for it in range(h+1):
            node_labels[it] = [0] * len(graph_list)

        for i, g in enumerate(graph_list):
            node_labels[0][i] = {(key, self._compress(value)) for key, value in nx.get_node_attributes(g, 'label').iteritems()}
            node_depth[i] = {(key, int(value)) for key, value in nx.get_node_attributes(g, 'depth').iteritems()}
            all_graphs_number_of_nodes += len([node for node in nx.node_iter(g) if node_depth[node] <= D])
            # _graph_to_dot(g, edge_labels[0][i], "graph{}.dot".format(i))

        # all_graphs_number_of_nodes is upper bound for number of possible edge labels
        phi = np.zeros((all_graphs_number_of_nodes, len(graph_list)), dtype=np.uint64)

        # h = 0
        for i, g in enumerate(graph_list):
            for node in g.nodes_iter():
                if node_depth[i][node] <= D:
                    label = node_labels[0][i][node]
                    phi[self._compress(label), i] += 1

        K = np.dot(phi.transpose(), phi)

        # h > 0
        for it in range(1, h+1):
            # Todo check if the shape fits in all cases
            phi = np.zeros((2*all_graphs_number_of_nodes, len(graph_list)))

            # for each graph update edge labels
            for i, g in enumerate(graph_list):
                node_labels[it][i] = {}
                for node in g.nodes_iter():
                    if node_depth[i][node] <= D:
                        label_collection = self._collect_labels(node, i, g, it, node_labels, node_depth, types, D)
                        long_label = "".join(np.concatenate([np.array([node_labels[it-1][i][node]]),
                                                             np.sort(label_collection)]))
                        node_labels[it][i][node] = self._compress(long_label)
                        phi[self._compress(long_label), i] += 1
                # _graph_to_dot(g, edge_labels[it][i], "graph{}_it{}.dot".format(i, it))

            K += np.dot(phi.transpose(), phi)

        return K




"""
Weisfeiler Lehman graph kernel for CFGs
"""
import numpy as np
import networkx as nx
from tqdm import tqdm
import time


class GK_WL(object):

    def __init__(self):
        self.__label_lookup__ = {}
        self.__label_counter__ = 0

    def _compress(self, long_label):
        if long_label not in self.__label_lookup__:
            self.__label_lookup__[long_label] = self.__label_counter__
            self.__label_counter__ += 1
        return self.__label_lookup__[long_label]

    def _collect_labels(self, node, i, graph, it, node_labels, node_depth, types, D, edge_types, edge_truth):
        ret = []
        for e in graph.in_edges_iter(nbunch=node, keys=True):
            # todo add for multigraph
            source, _, _ = e
            edge_t = edge_types[i][e]
            if edge_t in types and node_depth[i][source] <= D:
                long_edge_label = "_".join(
                        [str(t) for t in [node_labels[it][i][source],
                                          self._compress(str(edge_t)), self._compress(str(edge_truth[i][e]))]])
                # long_edge_label = "_".join(
                #         [str(t) for t in [node_labels[it][i][source],
                #                           str(edge_t), str(edge_truth[e])]])
                ret.append(self._compress(long_edge_label))
                # ret.append(long_edge_label)
        return ret

    @staticmethod
    def _graph_to_dot(graph, node_graph_labels, dot_file):
        g_copy = graph.copy()
        nx.set_node_attributes(g_copy, 'label', {k: str(v) for k, v in node_graph_labels.items()})
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
        edge_types = [0] * len(graph_list)
        edge_truth = [0] * len(graph_list)

        for it in range(h+1):
            node_labels[it] = [0] * len(graph_list)

        for i, g in enumerate(graph_list):
            node_labels[0][i] = {key: self._compress(value)
                                 for key, value in nx.get_node_attributes(g, 'label').items()}
            node_depth[i] = nx.get_node_attributes(g, 'depth')
            edge_types[i] = nx.get_edge_attributes(g, 'type')
            edge_truth[i] = nx.get_edge_attributes(g, 'truth')
            all_graphs_number_of_nodes += len([node for node in nx.nodes_iter(g) if node_depth[i][node] <= D])
            # if i == 0:
            #     self._graph_to_dot(g, node_labels[0][i], "graph{}.dot".format(i))

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
            phi = np.zeros((2*all_graphs_number_of_nodes, len(graph_list)), dtype=np.uint64)

            print('Updating node labels of graphs in iteration {}'.format(it), flush=True)

            # for each graph update edge labels
            for i, g in tqdm(list(enumerate(graph_list))):
                node_labels[it][i] = {}
                for node in g.nodes_iter():
                    if node_depth[i][node] <= D:
                        label_collection = self._collect_labels(node, i, g, it-1, node_labels, node_depth, types, D, edge_types, edge_truth)
                        long_label = "_".join(str(x) for x in [np.concatenate([np.array([node_labels[it-1][i][node]]),
                                                               np.sort(label_collection)])])
                        node_labels[it][i][node] = self._compress(long_label)
                        phi[self._compress(long_label), i] += 1
                        # node_labels[it][i][node] = long_label
                        # phi[self._compress(long_label), i] += 1
                # if i == 0:
                #     self._graph_to_dot(g, node_labels[it][i], "graph{}_it{}.dot".format(i, it))

            K += np.dot(phi.transpose(), phi)

        return K




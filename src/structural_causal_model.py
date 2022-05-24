from abc import ABC, abstractmethod
from pandas import DataFrame
from typing import List, Tuple, Dict


class StructuralCausalModel(ABC):
    def __init__(self, edge_list: List[Tuple[str, str, int or float]] = None, ):
        self.edges = []
        self.nodes = []
        self.weights = {}
        for (node_par, node_chi, w) in edge_list:
            self.edges.append((node_par, node_chi))
            if node_par not in self.nodes:
                self.nodes.append(node_par)
            if node_chi not in self.nodes:
                self.nodes.append(node_chi)
            self.weights[(node_par, node_chi)] = w
        self._check_info()
        self.adjacency_mtr = self._get_adjacency_mtr()
        self.adjacency_lst = self._get_adjacency_lst()

    def _check_info(self):
        # test whether the graph is acyclic
        res = set()
        for (a, b) in self.edges:
            if (a, b) and (b, a) not in res:
                res.add((a, b))
        temp_edges = list(res)
        if len(self.edges) != len(temp_edges):
            raise Exception("'edges' cannot contain both edges (a, b) and (b, a).")
        else:
            del res, temp_edges

    def _get_adjacency_mtr(self) -> DataFrame:
        adj_mtr = [[0 for i in range(len(self.nodes))] for j in range(len(self.nodes))]
        for edge in self.edges:
            i0 = self.nodes.index(edge[0])
            i1 = self.nodes.index(edge[1])
            # reads edge i0 -> i1: from the ith row to the jth column
            adj_mtr[i0][i1] = self.weights.get((edge[0], edge[1]))
        adj_mtr = DataFrame(adj_mtr, index=self.nodes, columns=self.nodes)
        return adj_mtr

    def _get_adjacency_lst(self) -> Dict[str, List[str]]:
        adj_lst = dict()
        for node in self.nodes:
            adj_lst[node] = list()
        for edge in self.edges:
            i0, i1 = edge[0], edge[1]
            if i1 not in adj_lst[i0]:
                adj_lst[i0].append(i1)
        return adj_lst

    @abstractmethod
    def define_sem(self):
        pass

    @abstractmethod
    def run_sem(self):
        pass

    @abstractmethod
    def generate_scfs(self):
        pass

    # @abstractmethod
    # def generate_scfs(self):
    #     pass

#
# EOF
#

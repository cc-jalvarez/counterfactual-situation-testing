from structural_causal_model import StructuralCausalModel as SCM
from typing import List, Tuple, Dict
from pandas import DataFrame


class LawSchool(SCM):
    def __init__(self,
                 list_edges_w_weights: List[Tuple[str, str, int or float]],
                 data: DataFrame = None, ):
        super(LawSchool, self).__init__(edge_list=list_edges_w_weights, )
        self.data = data

        # if self.adj_mtr is None:
        #     super().get_adj_mtr()
        # if self.adj_lst is None:
        #     super().get_adj_lst()




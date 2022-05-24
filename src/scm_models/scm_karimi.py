from scm_models.structural_causal_model import StructuralCausalModel as SCM
from typing import List, Tuple, Dict
from pandas import DataFrame


class KarimiPaper(SCM):
    def __init__(self,
                 list_edges_w_weights: List[Tuple[str, str, int or float]], ):
        super(KarimiPaper, self).__init__(edge_list=list_edges_w_weights, )

    def define_sem(self):
        pass

    def run_sem(self):
        pass

    def generate_scfs(self):
        pass


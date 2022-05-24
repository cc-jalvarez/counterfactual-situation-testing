from structural_causal_model import StructuralCausalModel as SCM
from typing import List, Tuple, Dict
from pandas import DataFrame


class LawSchool(SCM):
    def __init__(self,
                 list_edges_w_weights: List[Tuple[str, str, int or float]],
                 data: DataFrame = None, ):
        super(LawSchool, self).__init__(edge_list=list_edges_w_weights, )
        self.causal_sufficiency = causal_sufficiency
        self.SE = None
        self.data = data

    def get_structural_equations(self) -> Dict:
        """ Define your set of structural equations here """
        self.SE = {}
        print("introduce the structural equations as a dictionary of functions using 'SE'")
        # todo: seems hard to automate this part...


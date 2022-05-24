from structural_causal_model import StructuralCausalModel as SCM
from typing import List, Tuple, Dict
from pandas import DataFrame


class LawSchool(SCM):
    def __init__(self,
                 list_edges_w_weights: List[Tuple[str, str, int or float]],
                 end_vars: List[str],
                 exo_vars: List[str],
                 data: DataFrame = None, ):
        super(LawSchool, self).__init__(edge_list=list_edges_w_weights, )
        self.end_vars = end_vars
        self.exo_vars = exo_vars
        self.SEM = None
        self.data = data
        self.scfs = None

    def get_structural_equation_model(self) -> Dict:
        """ Define your set of structural equations as lambda row: df_var(row[x1],...,row[xj]) """
        if self.SEM:
            print("already have a structural equation model dict; overwrite it via 'SEM' if needed")
        else:
            self.SEM = dict.fromkeys(self.end_vars)
            print("introduce the structural equation model as a dict via 'SEM'")
            for key in self.SEM.keys():
                print(f"provide def_{key.lower()} function for {key}")
            print("provide each in the form: 'lambda row: df_var(row[x1],...,row[xj])'")

    def generate_fcts(self, data: DataFrame = None, update_data: bool = False) -> DataFrame:
        """ Here, define the order of equations for the factuals (FCTs) """
        if data is None:
            df = self.data
        else:
            df = data.copy()

        temp_new_vars = []

        print('generating FCTs in the following order:')
        for var in self.end_vars:
            print(var)
            df[f'fct_{var}'] = df.apply(self.SEM[var], axis=1)
            temp_new_vars.append(f'fct_{var}')
        print('generated the new variables:')
        print(*temp_new_vars, sep='\n')
        del temp_new_vars

        if update_data:
            self.data = df

        return df

    def generate_scfs(self, do: Dict, data: DataFrame = None, update_data: bool = False, ) -> DataFrame:
        """ Generate the structural counterfactuals (SCFs) using the SEM """
        if self.scfs is None:
            self.scfs = {}

        if data is None:
            df = self.data
        else:
            df = data.copy()

        # do: {'var_to_intervene': 'intervention'}
        
        if update_data:
            self.data = df

        return df


    # todo: generate model
    # todo: generate scf
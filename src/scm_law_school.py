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

    def define_sem(self) -> Dict:
        """ Define your set of structural equation model (SEM) as lambda row: df_var(row[x1],...,row[xj]) """
        if self.SEM:
            print("class instance already have a structural equation model dict; overwrite it via 'SEM' if needed")
        else:
            self.SEM = dict.fromkeys(self.end_vars)
            print("introduce the structural equation model as a dict via 'SEM'")
            for key in self.SEM.keys():
                print(f"provide def_{key.lower()} function for {key}")
            print("provide each in the form: 'lambda row: df_var(row[x1],...,row[xj])'")

    def run_sem(self, data: DataFrame = None, type: str = 'fct', update_data: bool = False) -> DataFrame:
        """ Here, define the order of equations for the SEM model: returns the factuals (FCTs) """
        if data is None:
            df = self.data
        else:
            df = data.copy()

        temp_new_vars = []

        print(f'generating {type.upper()}s in the following order:')
        for var in self.end_vars:
            print(var)
            df[f'{type}_{var}'] = df.apply(self.SEM[var], axis=1)
            temp_new_vars.append(f'{type}_{var}')
        print('generated the new variables:')
        print(*temp_new_vars, sep='\n')
        del temp_new_vars

        if update_data:
            self.data = df

        return df

    def generate_scfs(self, do: Dict, do_desc: str, data: DataFrame = None, ) -> DataFrame:
        """ Generate the structural counterfactuals (SCFs) using the SEM model"""
        if self.scfs is None:
            self.scfs = {}

        # track the intervened data
        if data is None:
            df = self.data
        else:
            df = data.copy()

        # do: {'var_to_intervene': 'intervention'}
        for do_var in do.keys():
            print(f'do({do_var}={do[do_var]}')
            df[f'org_{do_var}'] = df[do_var].copy()
            df[do_var] = do[do_var]
        # run SEM model on intervened data
        df = self.run_model(data=df, type='scf', update_data=False)

        self.scfs[do_desc] = df

        return df


    # todo: generate model
    # todo: generate scf
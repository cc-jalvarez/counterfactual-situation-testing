from scm_models.structural_causal_model import StructuralCausalModel as SCM
# from structural_causal_model import StructuralCausalModel as SCM
from typing import List, Tuple, Dict
from pandas import DataFrame


class LawSchool(SCM):
    def __init__(self,
                 list_edges_w_weights: List[Tuple[str, str, int or float]],
                 end_vars: List[str],
                 exo_vars: List[str] = None,
                 data: DataFrame = None, ):
        super(LawSchool, self).__init__(edge_list=list_edges_w_weights, )
        self.end_vars = end_vars
        self.exo_vars = exo_vars
        self.data = data
        self.SEM = None
        self.scfs = None

    def define_sem(self):
        """ Define your set of structural equation model (SEM) as lambda row: df_var(row[x1],...,row[xj]) """
        if self.SEM:
            print("class instance already has a structural equation model dict; overwrite it via 'SEM' if needed")
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

    def generate_scfs(self, do: Dict[str, int or float], do_desc: str, data: DataFrame = None, ) -> DataFrame:
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
            print(f'do({do_var}={do[do_var]})')
            df[f'org_{do_var}'] = df[do_var].copy()
            df[do_var] = do[do_var]

        # run SEM model on intervened data
        df = self.run_sem(data=df, type='scf', update_data=False)

        # store the counterfactuals by do-type
        self.scfs[do_desc] = df

        return df

#
# EOF
#

# # define the DAG
# dag_law_school = [('U', 'UGPA', ugpa_weights.loc[0, 'eta_u_ugpa']),
#                   ('U', 'LSAT', lsat_weights.loc[0, 'eta_u_lsat']),
#                   ('female', 'UGPA', ugpa_weights.loc[0, 'female']),
#                   ('male', 'UGPA', ugpa_weights.loc[0, 'male']),
#                   ('white', 'UGPA', ugpa_weights.loc[0, 'white']),
#                   ('nonwhite', 'UGPA', ugpa_weights.loc[0, 'nonwhite']),
#                   ('female', 'LSAT', lsat_weights.loc[0, 'female']),
#                   ('male', 'LSAT', lsat_weights.loc[0, 'male']),
#                   ('white', 'LSAT', lsat_weights.loc[0, 'white']),
#                   ('nonwhite', 'LSAT', lsat_weights.loc[0, 'nonwhite'])
#                  ]
#
# # initiate the class LawSchool to get all the SCM methods (maybe too much for this...)
# law_school = LawSchool(dag_law_school,
#                        end_vars=['UGPA', 'LSAT'],
#                        exo_vars=['U'], )
#
# # it includes some nice methods
# print(law_school.nodes)
# print(law_school.weights)
# print(law_school.adjacency_mtr)
# print(law_school.adjacency_lst)
#
# law_school.define_sem()
#
# # UGPA
# def pred_ugpa(v_u, v_female, v_male, v_white, v_nonwhite):
#     return (ugpa_weights.loc[0, 'ugpa0'] +
#             law_school.adjacency_mtr.loc['U']['UGPA'] * v_u +
#             law_school.adjacency_mtr.loc['female']['UGPA'] * v_female +
#             law_school.adjacency_mtr.loc['male']['UGPA'] * v_male +
#             law_school.adjacency_mtr.loc['white']['UGPA'] * v_white +
#             law_school.adjacency_mtr.loc['nonwhite']['UGPA'] * v_nonwhite)
#
#
# # LSAT
# def pred_lsat(v_u, v_female, v_male, v_white, v_nonwhite):
#     return np.exp(lsat_weights.loc[0, 'lsat0'] +
#                   law_school.adjacency_mtr.loc['U']['LSAT'] * v_u +
#                   law_school.adjacency_mtr.loc['female']['LSAT'] * v_female +
#                   law_school.adjacency_mtr.loc['male']['LSAT'] * v_male +
#                   law_school.adjacency_mtr.loc['white']['LSAT'] * v_white +
#                   law_school.adjacency_mtr.loc['nonwhite']['LSAT'] * v_nonwhite)
#
# law_school.SEM['UGPA'] = lambda row: pred_ugpa(
#     v_u=row['U'], v_female=row['female'], v_male=row['male'], v_white=row['white'], v_nonwhite=row['nonwhite'])
#
# law_school.SEM['LSAT'] = lambda row: pred_lsat(
#     v_u=row['U'], v_female=row['female'], v_male=row['male'], v_white=row['white'], v_nonwhite=row['nonwhite'])
#
# law_school.define_sem()
#
# do_male = law_school.generate_scfs(do={'female': 0, 'male': 1},
#                                    do_desc='do_male',
#                                    data=df)
# do_male.head(5)
#
# do_white = law_school.generate_scfs(do={'nonwhite': 0, 'white': 1},
#                                     do_desc='do_white',
#                                     data=df)
# do_white.head(5)

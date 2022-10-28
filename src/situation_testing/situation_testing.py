from typing import List, Dict, Tuple
from pandas import DataFrame
import heapq
import math
import pandas as pd
import numpy as np

# local
from _distance_functions import kdd2011dist
from _utils import *

__DISTANCES__ = {'kdd2011': kdd2011dist}


class SituationTesting:
    def __init__(self,
                 df: DataFrame, cf_df: DataFrame = None,
                 nominal_atts: List[str] = None, continuous_atts: List[str] = None, ordinal_atts: List[str] = None,
                 *args, **kwargs
                 ):

        # standard k-NN ST vs counterfactual k-NN ST
        self.df = df  # todo: why feed the data at all here?
        self.cf_df = cf_df  # todo: should we check the two?
        # to avoid mutable objects as default arguments
        nominal_atts = [] if nominal_atts is None else nominal_atts
        continuous_atts = [] if continuous_atts is None else continuous_atts
        ordinal_atts = [] if ordinal_atts is None else ordinal_atts

        # set class baseline parameters
        self.nominal_atts = nominal_atts
        self.continuous_atts = continuous_atts
        self.ordinal_atts = ordinal_atts
        self.all_atts = {'nominal_atts': self.nominal_atts, 'continuous_atts': self.nominal_atts, 'ordinal_atts': self.ordinal_atts}
        self.relevant_atts = self.nominal_atts + self.continuous_atts + self.ordinal_atts
        print('relevant attributes: {rel_att}'.format(rel_att=self.relevant_atts))

        # todo
        # # store positions of attributes
        # cols = list(df.columns)  # cf_df by construction has the same columns as df
        # self.nominal_atts_pos = [cols.index(c) for c in nominal_atts]
        # self.continuous_atts_pos = [cols.index(c) for c in continuous_atts]
        # self.ordinal_atts_pos = [cols.index(c) for c in ordinal_atts]
        # self.natts = len(continuous_atts) + len(nominal_atts) + len(ordinal_atts)
        # # statistics of continuous features
        # self.means = {c: df[c].mean() for c in continuous_atts}
        # self.stds = {c: df[c].std() for c in continuous_atts}
        # # statistics of ordinal features
        # self.nofvalues = {c: (df[c].nunique() - 1) for c in ordinal_atts}
        # # positions of features (for future usage)
        # cols = list(df.columns)
        #
        # self.stds_pos = {self.continuous_atts_pos[i]: self.stds[c] for i, c in enumerate(continuous_atts)}
        # self.nofvalues_pos = {self.ordinal_atts_pos[i]: self.nofvalues[c] for i, c in enumerate(ordinal_atts)}

    def top_k(self, t, tset, k: int, distance: str, max_d: float = None) -> List[Tuple[float, int]]:
        """
        Parameters:

        Returns:
        list of pairs: list of (distance, index) of the k closest instances to t at a distance of at most max_d
        """
        ds = __DISTANCES__[distance](t, tset, self.all_atts)
        q = []
        lenq = 0
        for i, d in zip(tset.index, ds):
            if max_d is None or d <= max_d:
                if lenq < k:
                    heapq.heappush(q, (-d, i))
                    lenq += 1
                else:
                    d1, _ = heapq.heappushpop(q, (-d, i))
                    max_d = -d1
        q = [(-v, i) for v, i in q]
        return sorted(q)

    # TODO
    # def get_search_spaces(self, sensitive_att: str or List[str], sensitive_val: Dict):
    #     if isinstance(sensitive_att, list):  # multiple or intersectional disc.
    #         if len(sensitive_att) == len(sensitive_val):
    #             print('multiple disc: consider A_1 + A+2 + ... + A_n')
    #         else:
    #             print('intersectional disc: consider A_1 * A+2 * ... * A_n')
    #
    #     else:  # single disc.
    #         search_me = self.df[self.df[sensitive_att] == get_pro_value(sensitive_val)].copy()
    #
    #     pass

    def get_k_neighbors(self,
                        target_att: str, target_val: Dict,
                        sensitive_att: str or List[str], sensitive_val: Dict,
                        k: int,
                        distance: str = 'kdd2011', max_d: float = None,
                        relevant_atts: List[str] = None,
                        return_counterfactual_fairness: bool = False,
                        return_neighbors: bool = False):

        # output
        res_st = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        # additional outputs
        if return_neighbors:
            dict_df_neighbors = {}
        else:
            dict_df_neighbors = None
        if return_counterfactual_fairness:
            res_cf = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        else:
            res_cf = None

        # consider only the relevant decision-making attributes: don't include A nor y
        self.relevant_atts = relevant_atts if relevant_atts else self.relevant_atts
        # gather info for control (ctr) and test (tst) groups
        # if isinstance(sensitive_att, list): # todo: atm, only |A|=1
        #     print('multiple/intersectional discrimination')
        # else:
        bad_y_val = get_neg_value(target_val)
        sensitive_val = get_pro_value(sensitive_val)
        ctr_set = self.df[sensitive_att] == sensitive_val  # returns a pd.Series of booleans
        ctr_idx = self.df[ctr_set].index.to_list()
        tst_idx = self.df[~ctr_set].index.to_list()
        # sensitive_att = [sensitive_att]  # todo: turn it into a list to follow dd loop
        # define the search spaces
        ctr_search = self.df[ctr_set].copy()
        if self.cf_df:
            print('running counterfactual ST search')
            tst_search = self.cf_df[~ctr_set].copy()
        else:
            print('running standard ST search')
            tst_search = self.df[~ctr_search].copy()
        # find k-control and k-test neighbors
        # for A in sensitive_att: TODO
        for i, row in ctr_search[self.relevant_atts].iterrows():
            ctr_k = self.top_k(row, ctr_search[self.relevant_atts], k + 1, distance, max_d)
            tst_k = self.top_k(row, tst_search[self.relevant_atts], k, distance, max_d)  # todo: need the CF!
            nn1 = [j for _, j in ctr_k if j != i]  # idx for ctr_k (minus center)
            nn2 = [j for _, j in tst_k]            # idx for tst_k
            p1 = sum(self.df.loc[nn1, ][target_att] == bad_y_val) / len(nn1)
            p2 = sum(self.cf_df.loc[nn2, ][target_att] == bad_y_val) / len(nn2) # todo can queery df here!!!
            res_st.loc[i] = round(p1 - p2, 3)
            if dict_df_neighbors:
                i_dict_df_neighbors = {'ctr_idx': nn1, 'tst_idx': nn2}
                dict_df_neighbors[int(i)] = i_dict_df_neighbors
            if res_cf:
                if self.df.loc[i, target_att] == self.cf_df.loc[i, target_att] == bad_y_val:
                    res_cf[i] = 'Yes'
                else:
                    res_cf[i] = 'No'

        return res_st, dict_df_neighbors, res_cf

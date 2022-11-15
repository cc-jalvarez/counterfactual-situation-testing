from typing import List, Dict, Tuple
from pandas import DataFrame
from sklearn import preprocessing
import heapq
import math
import pandas as pd
import numpy as np
import scipy.stats as st
# local functions
from situation_testing._distance_functions import kdd2011dist
from situation_testing._utils import *


__DISTANCES__ = {'kdd2011': kdd2011dist}


class SituationTesting:
    def __init__(self, *args, **kwargs):
        self.df = None
        self.cf_df = None
        self.nominal_atts = None
        self.continuous_atts = None
        self.ordinal_atts = None
        self.all_atts = None
        self.relevant_atts = None
        self.nominal_atts_pos = None
        self.continuous_atts_pos = None
        self.ordinal_atts_pos = None
        self.natts = None
        self.res_dict_df_neighbors = None
        self.res_cf = None
        self.wald_ci = None

    def setup_baseline(self, df: DataFrame,
                       cf_df: DataFrame = None,
                       nominal_atts: List[str] = None,
                       continuous_atts: List[str] = None,
                       ordinal_atts: List[str] = None):
        self.df = df
        self.cf_df = cf_df
        # all attribute information
        nominal_atts = [] if nominal_atts is None else nominal_atts
        continuous_atts = [] if continuous_atts is None else continuous_atts
        ordinal_atts = [] if ordinal_atts is None else ordinal_atts
        self.nominal_atts = nominal_atts
        self.continuous_atts = continuous_atts
        self.ordinal_atts = ordinal_atts
        self.relevant_atts = self.nominal_atts + self.continuous_atts + self.ordinal_atts
        self.all_atts = {'nominal_atts': self.nominal_atts,
                         'continuous_atts': self.continuous_atts,
                         'ordinal_atts': self.ordinal_atts}
        cols = list(df.columns)
        self.nominal_atts_pos = [cols.index(c) for c in nominal_atts]
        self.continuous_atts_pos = [cols.index(c) for c in continuous_atts]
        self.ordinal_atts_pos = [cols.index(c) for c in ordinal_atts]
        self.natts = len(continuous_atts) + len(nominal_atts) + len(ordinal_atts)
        # normalize the data
        scaler = preprocessing.StandardScaler()
        print('standardizing factual dataset')
        self.df[self.continuous_atts] = scaler.fit_transform(self.df[self.continuous_atts])
        if self.cf_df is not None:
            print('standardizing counterfactual dataset')
            self.cf_df[self.continuous_atts] = scaler.fit_transform(self.cf_df[self.continuous_atts])

    def top_k(self, t, tset, k: int, distance: str, max_d: float = None) -> List[Tuple[float, int]]:
        """
        Parameters:

        Returns:
        list of pairs: list of (distance, index) of the k closest instances to t at a distance of at most max_d
        """
        ds = __DISTANCES__[distance](t, tset, self.relevant_atts, self.all_atts)
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

    def run(self, target_att: str, target_val: Dict, sensitive_att: str or List[str], sensitive_val: Dict, k: int,
            alpha: float = 0.05,
            tau: float = 0.0,
            distance: str = 'kdd2011',
            max_d: float = None,
            return_counterfactual_fairness: bool = True,
            return_neighbors: bool = True):

        # outputs:
        res_st = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        self.wald_ci = []
        # other outputs:
        if return_neighbors:
            self.res_dict_df_neighbors = {}
        if return_counterfactual_fairness and self.cf_df is not None:
            self.res_cf = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        else:
            return_counterfactual_fairness = False

        # todo: atm, only |A|=1
        # gather info for control (ctr) and test (tst) groups
        # if isinstance(sensitive_att, list):
        #     print('multiple/intersectional discrimination')
        # else:
        # todo: turn it into a list to follow dd loop
        # sensitive_att = [sensitive_att]

        # update relevant attribute list: exclude target and sensitive att(s)
        self.relevant_atts = [att for att in self.relevant_atts if att not in sensitive_att or att == target_att]
        bad_y_val = get_neg_value(target_val)
        sensitive_val = get_pro_value(sensitive_val)
        sensitive_set = self.df[sensitive_att] == sensitive_val  # returns a pd.Series of booleans
        # define search spaces
        ctr_search = self.df.loc[sensitive_set, self.relevant_atts].copy()
        tst_search = self.df.loc[~sensitive_set, self.relevant_atts].copy()
        # find idx for control and test neighborhoods
        for c in self.df[sensitive_set].index.to_list():
            ctr_k = self.top_k(self.df.loc[c, self.relevant_atts], ctr_search, k + 1, distance, max_d)
            if self.cf_df is not None:
                # cfST: draw test center from counterfactual df
                tst_k = self.top_k(self.cf_df.loc[c, self.relevant_atts], tst_search, k, distance, max_d)
            else:
                # standard ST: draw test center from factual df
                tst_k = self.top_k(self.df.loc[c, self.relevant_atts], tst_search, k, distance, max_d)
            nn1 = [j for _, j in ctr_k if j != c]  # idx for ctr_k (minus center)
            nn2 = [j for _, j in tst_k]            # idx for tst_k
            k1 = len(nn1)
            k2 = len(nn2)
            p1 = sum(self.df.loc[nn1, target_att] == bad_y_val) / k1
            p2 = sum(self.df.loc[nn2, target_att] == bad_y_val) / k2
            # output(s)
            res_st.loc[c] = round(p1 - p2, 3)  # diff
            self._test_discrimination(c, p1, p2, k1, k2, alpha, tau)  # statistical diff
            if return_neighbors:
                self.res_dict_df_neighbors[int(c)] = {'ctr_idx': nn1, 'tst_idx': nn2}
            if return_counterfactual_fairness:
                if self.df.loc[c, target_att] == self.cf_df.loc[c, target_att]:  # TODO: only for protected ind!
                    self.res_cf[c] = True
                else:
                    self.res_cf[c] = False

        return res_st

    def _test_discrimination(self, ind, p1, p2, k1, k2, alpha, tau, sigfig: int = 3):
        z_score = round(st.norm.ppf(1 - (alpha / 2)), sigfig)
        d_alpha = z_score * math.sqrt((p1 * (1 - p1) / k1) + (p2 * (1 - p2) / k2))
        conf_inter = [round((p1 - p2) - d_alpha, sigfig), round((p1 - p2) + d_alpha, sigfig)]
        org_diff = round(p1 - p2, sigfig)
        if (p1 - p2) >= 0:  # from ST paper #1
            diff = round(max(0, p1 - p2 - d_alpha), sigfig)
        else:
            diff = round(min(0, p1 - p2 + d_alpha), sigfig)
        # discrimination evidence:
        if org_diff > tau:
            cf_st = 'Yes'
        else:
            cf_st = 'No'
        self.wald_ci.append(
            {
                'individual': ind,
                'p1': p1,
                'p2': p2,
                'org_diff': org_diff,
                'd_alpha': d_alpha,
                'diff': diff,
                'CIs': conf_inter,
                'cfST': cf_st
            }
        )

    def get_test_discrimination(self):
        return pd.DataFrame(self.wald_ci)

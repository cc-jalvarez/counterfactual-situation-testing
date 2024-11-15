from typing import List, Dict, Tuple
from pandas import DataFrame
from sklearn import preprocessing
import heapq
import math
import pandas as pd
import numpy as np
import scipy.stats as st
# from src.situation_testing._distance_functions import kdd2011dist
# from src.situation_testing._utils import *
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
        self.normalize = None
        self.natts = None
        self.include_centers = None
        self.res_dict_df_neighbors = None
        self.res_dict_dist_to_neighbors = None
        self.res_counterfactual_unfairness = None
        self.wald_ci = None

    def setup_baseline(self, df: DataFrame, cf_df: DataFrame = None, nominal_atts: List[str] = None,
                       continuous_atts: List[str] = None, ordinal_atts: List[str] = None, normalize: bool = True):
        # datasets
        self.df = df
        self.cf_df = cf_df
        # all attributes information
        nominal_atts = [] if nominal_atts is None else nominal_atts
        continuous_atts = [] if continuous_atts is None else continuous_atts
        ordinal_atts = [] if ordinal_atts is None else ordinal_atts
        self.nominal_atts = nominal_atts
        self.continuous_atts = continuous_atts
        self.ordinal_atts = ordinal_atts
        self.relevant_atts = self.nominal_atts + self.continuous_atts + self.ordinal_atts
        self.normalize = normalize
        self.all_atts = {'nominal_atts': self.nominal_atts,
                         'continuous_atts': self.continuous_atts,
                         'ordinal_atts': self.ordinal_atts,
                         'normalize': self.normalize}
        cols = list(df.columns)
        self.nominal_atts_pos = [cols.index(c) for c in nominal_atts]
        self.continuous_atts_pos = [cols.index(c) for c in continuous_atts]
        self.ordinal_atts_pos = [cols.index(c) for c in ordinal_atts]
        self.natts = len(continuous_atts) + len(nominal_atts) + len(ordinal_atts)
        # normalize the data if normalize is True
        if self.normalize:
            scaler = preprocessing.StandardScaler()
            self.df[self.continuous_atts] = scaler.fit_transform(self.df[self.continuous_atts])
            if self.cf_df is not None:
                self.cf_df[self.continuous_atts] = scaler.fit_transform(self.cf_df[self.continuous_atts])

    def top_k(self, t, tset, k: int, distance: str, max_d: float = None) -> List[Tuple[float, int]]:
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

    # TODO: are we using this at all?
    def run_mul(self, target_att: str, target_val: Dict, sensitive_att: List[str], sensitive_val: List[Dict],
                k: int,
                alpha: float = 0.05,
                tau: float = 0.0,
                distance: str = 'kdd2011',
                max_d: float = None,
                include_centers: bool = None,
                return_counterfactual_fairness: bool = True) -> DataFrame:
        res_mul = self.df[[sensitive_att]].copy()
        for a in range(len(sensitive_att)):
            res_mul['diff_' + sensitive_att[a]] = self.run(target_att, target_val, sensitive_att[a], sensitive_val[a],
                                                           k, alpha, tau, distance, max_d, include_centers, False)
            return res_mul

    def run(self, target_att: str, target_val: Dict, sensitive_att: str, sensitive_val: Dict, k: int,
            alpha: float = 0.05, tau: float = 0.0, distance: str = 'kdd2011', max_d: float = None,
            include_centers: bool = None, return_counterfactual_fairness: bool = True, sigfig: int = 3):

        # when True, include ctr and tst centers in p1 (ctr group) and p2 (tst group) calculations
        self.include_centers = include_centers if include_centers is not None else self.include_centers
        # outputs:
        res_st = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        self.wald_ci = []
        self.res_dict_df_neighbors = {}
        self.res_dict_dist_to_neighbors = {}
        # when True, output counterfactual fairness
        if return_counterfactual_fairness and self.cf_df is not None:
            self.res_counterfactual_unfairness = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        else:
            return_counterfactual_fairness = False

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
                # CST: draw test search center from counterfactual df
                if self.include_centers:
                    temp_tst_search = tst_search.copy()
                    temp_tst_search = temp_tst_search.append(self.cf_df.loc[c, self.relevant_atts])
                    # CST w/
                    tst_k = self.top_k(self.cf_df.loc[c, self.relevant_atts], temp_tst_search, k + 1, distance, max_d)
                    del temp_tst_search
                else:
                    # CST w/o
                    tst_k = self.top_k(self.cf_df.loc[c, self.relevant_atts], tst_search, k, distance, max_d)
            else:
                # ST: draw test search center from factual df
                tst_k = self.top_k(self.df.loc[c, self.relevant_atts], tst_search, k, distance, max_d)
            if self.cf_df is not None and self.include_centers:
                # running CST w/
                nn1 = [j for _, j in ctr_k]
                nn2 = [j for _, j in tst_k]
                k1 = len(nn1)
                k2 = len(nn2)
                p1 = sum(self.df.loc[nn1, target_att] == bad_y_val) / k1     # control
                p2 = sum(self.cf_df.loc[nn2, target_att] == bad_y_val) / k2  # test
            else:
                # running ST or CST w/o
                nn1 = [j for _, j in ctr_k if j != c]  # idx for ctr_k (minus center)
                nn2 = [j for _, j in tst_k]            # idx for tst_k (not necessary here: only added for CST w/)
                k1 = len(nn1)
                k2 = len(nn2)
                p1 = sum(self.df.loc[nn1, target_att] == bad_y_val) / k1  # control
                p2 = sum(self.df.loc[nn2, target_att] == bad_y_val) / k2  # test
            # output(s)
            res_st.loc[c] = round(p1 - p2, sigfig)  # delta
            self._test_discrimination(c, p1, p2, k1, k2, alpha, tau)  # statistical tests (includes delta)
            # return neighbors info:
            self.res_dict_df_neighbors[int(c)] = {'ctr_idx': [i for i in nn1 if i != c],
                                                  'tst_idx': [i for i in nn2 if i != c]}
            self.res_dict_dist_to_neighbors[int(c)] = {'ctr_idx': [d[0] for d in ctr_k if d[1] != c],
                                                       'tst_idx': [d[0] for d in tst_k if d[1] != c]}
            # counterfactual fairness info:
            if return_counterfactual_fairness:
                if self.df.loc[c, target_att] != self.cf_df.loc[c, target_att]:
                    # discrimination: neg_y to pos_y
                    if self.df.loc[c, target_att] == bad_y_val:
                        self.res_counterfactual_unfairness[c] = 1
                    # positive discrimination: pos_y to neg_y  TODO: revise this notion | unclear to me
                    if self.df.loc[c, target_att] != bad_y_val:
                        self.res_counterfactual_unfairness[c] = 2
                else:
                    self.res_counterfactual_unfairness[c] = 0

        return res_st

    def _test_discrimination(self, ind, p1, p2, k1, k2, alpha, tau, sigfig: int = 3, ngtv_disc: bool = True):
        # the point estimate
        delta_p = p1 - p2
        # one-sided test: used for ST and CST
        z_score_1 = round(st.norm.ppf(1 - alpha), sigfig)
        d_alpha_1 = z_score_1 * math.sqrt((p1 * (1 - p1) / k1) + (p2 * (1 - p2) / k2))
        # negative discrimination
        if ngtv_disc:
            # evidence for discrimination?
            if round(delta_p, sigfig) > tau:
                disc_evi = 'Yes'
            else:
                disc_evi = 'No'
            # statistically significant evidence?
            ci_1 = round(delta_p - d_alpha_1, sigfig)  # round(min(0, p1 - p2 - d_alpha), sigfig)
            if tau >= ci_1:
                stat_evi = 'No'
            else:
                stat_evi = 'Yes'
        # positive discrimination
        else:
            # evidence for discrimination?
            if round(delta_p, sigfig) < tau:
                disc_evi = 'Yes'
            else:
                disc_evi = 'No'
            # statistically significant evidence?
            ci_1 = round(delta_p + d_alpha_1, sigfig)  # round(max(0, p1 - p2 + d_alpha), sigfig)
            if tau <= ci_1:
                stat_evi = 'No'
            else:
                stat_evi = 'Yes'
        # two-sided test: used for confidence in CF
        z_score_2 = round(st.norm.ppf(1 - alpha/2), sigfig)
        d_alpha_2 = z_score_2 * math.sqrt((p1 * (1 - p1) / k1) + (p2 * (1 - p2) / k2))
        ci_2 = [round(delta_p - d_alpha_2, sigfig), round(delta_p + d_alpha_2, sigfig)]

        # TODO: check with Salvatore before implementing
        # if (p1 - p2) >= 0:
        #     diff = round(max(0, p1 - p2 - d_alpha), sigfig)
        # else:
        #     diff = round(min(0, p1 - p2 + d_alpha), sigfig)

        self.wald_ci.append(
            {
                'individual': ind,
                'p_c': p1,
                'p_t': p2,
                'delta_p': delta_p,
                'CI_1st': ci_1,
                'CI_2st': ci_2,
                'DiscEvi': disc_evi,
                'StatEvi': stat_evi
            }
        )

    def get_test_discrimination(self):
        return pd.DataFrame(self.wald_ci)

#
# EOF
#

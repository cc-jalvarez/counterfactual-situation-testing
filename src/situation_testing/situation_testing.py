
from pandas import DataFrame
from typing import List, Dict, Tuple
import heapq
import math

# local
from _distance_functions import kdd2011dist

__DISTANCES__ = {'kdd2011': kdd2011dist}  # todo: can expand here for other d's with same format


class SituationTesting:
    def __init__(self, df: DataFrame, cf_df: DataFrame = None,
                 nominal_atts: List[str] = None, continuous_atts: List[str] = None, ordinal_atts: List[str] = None):

        # standard k-NN ST vs counterfactual k-NN ST
        if cf_df:
            # todo: check that df and cf_df match
            self.cf_df = cf_df

        # to avoid mutable objects as default arguments
        nominal_atts = [] if nominal_atts is None else nominal_atts
        continuous_atts = [] if continuous_atts is None else continuous_atts
        ordinal_atts = [] if ordinal_atts is None else ordinal_atts

        # set class baseline parameters
        self.nominal_atts = nominal_atts
        self.continuous_atts = continuous_atts
        self.ordinal_atts = ordinal_atts
        self.all_atts = {'nominal_atts': self.nominal_atts,
                         'continuous_atts': self.nominal_atts,
                         'ordinal_atts': self.ordinal_atts}
        # store positions of attributes
        cols = list(df.columns)  # cf_df by construction has the same columns
        self.nominal_atts_pos = [cols.index(c) for c in nominal_atts]
        self.continuous_atts_pos = [cols.index(c) for c in continuous_atts]
        self.ordinal_atts_pos = [cols.index(c) for c in ordinal_atts]

        # todo
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

    def top_k(self, t, tset, distance: str, k: int, max_d: float = None) -> List[Tuple[float, int]]:
        # get distance function
        ds = __DISTANCES__[distance](t, tset, self.all_atts)  # todo: or add __DISTANCES__ to class and call via self
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





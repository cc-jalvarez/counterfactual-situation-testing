from typing import List, Dict
from pandas import DataFrame, Series
import math
import pandas as pd
import numpy as np


def kdd2011dist(t: Dict, tset: DataFrame, relevant_atts: List[str], atts_types: Dict[str, List[str]]) -> Series:

    # todo: relax the following assumptions
    # assumes distance estimation for one-to-many tuples
    # assumes t is in tset
    # assumes continuous attributes have been normalized

    tot = pd.Series(np.zeros(len(tset)), index=tset.index)
    # use normalized Manhattan distance for continuous and ordinal attributes; use overlap measurement for nominal
    for c in relevant_atts:
        if c in atts_types['continuous_atts']:
    # for c in atts_types['continuous_atts']:
            dist = abs(t[c] - tset[c]) / (max(tset[c]) - min(tset[c]))
            tot += dist
        if c in atts_types['ordinal_atts']:
    # for c in atts_types['ordinal_atts']:
            n_vals = tset[c].nunique() - 1
            val = t[c] / n_vals
            tmp = tset[c] / n_vals
            if math.isnan(val):
                dist = max(tmp, 1 - tmp)
                dist[dist.isnull()] = 1
            else:
                dist = abs(val - tmp)
                dist[dist.isnull()] = max(val, 1 - val)
            tot += dist
        if c in atts_types['nominal_atts']:
    # for c in atts_types['nominal_atts']:
            tot += 1 * (t[c] != tset[c])  # notice t[c]!=tset[c] is True if one or both are NaN

    # number of attributes to get final distance between tuples
    n_atts = len(atts_types['continuous_atts']) + len(atts_types['ordinal_atts']) + len(atts_types['nominal_atts'])

    return tot.divide(n_atts)


def manhattan(t: Dict, tset: DataFrame, atts_types: Dict[str, List[str]]) -> Series:
    tot = pd.Series(np.zeros(len(tset)), index=tset.index)
    for c in tset.columns:
        dist = abs(t[c] - tset[c]) / (max(tset[c]) - min(tset[c]))
        tot += dist
    n_atts = len(atts_types['continuous_atts']) + len(atts_types['ordinal_atts']) + len(atts_types['nominal_atts'])
    return tot.divide(n_atts)

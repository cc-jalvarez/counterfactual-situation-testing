import os
import pandas as pd
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import List, Dict
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from pandas import DataFrame
from tqdm import tqdm as progress_bar


def run_cf_situation_testing(
        df: DataFrame, cf_df: DataFrame, k: int,
        feat_trgt: str, feat_trgt_vals: Dict,
        feat_rlvt: List[str],
        feat_prot: str, feat_prot_vals: Dict,
        d: str = 'manhattan',  standardize: bool = False, weights: Dict = None,
        alpha: float = 0.05, tau: float = 0.05,
                             ) -> DataFrame:

    dict_k_neighbors = get_k_neighbors(df, cf_df, k,
                                       feat_trgt, feat_trgt_vals, feat_rlvt, feat_prot, feat_prot_vals,
                                       d, standardize, weights, )

    results_cf_st = get_wald_ci(dict_k_neighbors, feat_trgt, feat_trgt_vals, alpha, tau)

    return results_cf_st


def get_k_neighbors(df: DataFrame, cf_df: DataFrame,
                    k: int,
                    feat_trgt: str, feat_trgt_vals: Dict, feat_rlvt: List[str], feat_prot: str, feat_prot_vals: Dict,
                    d: str = 'manhattan', standardize: bool = False, weights: Dict = None,
                    ) -> Dict[int, Dict[str, DataFrame]]:

    # output:
    dict_df_neighbors = {}
    # input(s):
    feat_list = [feat_trgt] + feat_rlvt + [feat_prot]
    print(f"target feature {feat_trgt} with values {feat_trgt_vals}")
    print(f"protected feature {feat_prot} with values {feat_prot_vals}")
    print(f"with relevant features {feat_rlvt}")
    print(f"all features: {feat_list}")

    # individuals have the same index across df and cf_df
    protected_indices = df[df[feat_prot] == feat_prot_vals['protected']].index.to_list()
    non_protected_indices = df[df[feat_prot] == feat_prot_vals['non_protected']].index.to_list()

    # use factual df for ctr search space
    search_ctr_group = df[feat_rlvt].copy()
    # use counterfactual df for tst search space
    search_tst_group = cf_df[feat_rlvt].copy()

    if standardize:
        print('===> standardizing')

        # define standard-normal scaler
        scaler = preprocessing.StandardScaler()

        # ctr
        search_ctr_group_scaled = scaler.fit_transform(search_ctr_group)
        search_ctr_group_scaled = pd.DataFrame(search_ctr_group_scaled,
                                               index=search_ctr_group.index,
                                               columns=search_ctr_group.columns)
        search_ctr_group = search_ctr_group_scaled
        del search_ctr_group_scaled

        # tst
        search_tst_group_scaled = scaler.fit_transform(search_tst_group)
        search_tst_group_scaled = pd.DataFrame(search_tst_group_scaled,
                                               index=search_tst_group.index,
                                               columns=search_tst_group.columns)
        search_tst_group = search_tst_group_scaled
        del search_tst_group_scaled

    if weights:
        print('===> weighting')

        if len(weights) != len(feat_rlvt):
            sys.exit('provide a weight for each relevant feature')

        for feat_weight in weights:
            print(feat_weight)
            search_ctr_group[feat_weight] = weights[feat_weight] * search_ctr_group[feat_weight]
            search_tst_group[feat_weight] = weights[feat_weight] * search_tst_group[feat_weight]

    # define ctr centers (search_ctr_group will always include the ctr center)
    centers_ctr = search_ctr_group.iloc[protected_indices].copy()
    # update ctr search
    search_ctr_group = search_ctr_group.iloc[protected_indices].copy()
    search_ctr_group.reset_index(inplace=True, )
    search_ctr_group.rename(columns={'index': 'org_index'}, inplace=True)

    # define tst centers; tst search is updated within loop!
    centers_tst = search_tst_group.iloc[protected_indices].copy()

    for ind in protected_indices:

        # for ind's storing the neighbors
        temp_dict_df_neighbors = {}

        # get ctr center from df of factual centers
        ind_center_ctr = centers_ctr.loc[ind, ]  # [ind, feat_rlvt]
        # get tst center from df of counterfactual centers
        ind_center_tst = centers_tst.loc[ind, ]  # [ind, feat_rlvt]
        # prepare for knn
        if len(feat_rlvt) > 1:
            ind_center_ctr = ind_center_ctr.values.reshape(1, -1)
            ind_center_tst = ind_center_tst.values.reshape(1, -1)
        else:
            ind_center_ctr = ind_center_ctr.values.reshape(-1, 1)
            ind_center_tst = ind_center_tst.values.reshape(-1, 1)

        # ctr neighborhood
        knn_1 = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric=d).fit(search_ctr_group[feat_rlvt])
        distances_1, indices_1 = knn_1.kneighbors(ind_center_ctr)

        temp_ctr_df = pd.DataFrame()
        temp_ctr_df['knn_indices'] = pd.Series(indices_1[0])
        temp_ctr_df['knn_distances'] = pd.Series(distances_1[0])
        temp_ctr_df.sort_values(by='knn_distances', ascending=True, inplace=True)
        temp_ctr_df = temp_ctr_df.merge(search_ctr_group[['org_index']], how='inner', left_on='knn_indices', right_index=True)
        temp_ctr_df = temp_ctr_df.merge(df[feat_list], how='inner', left_on='org_index', right_index=True)
        # store
        temp_dict_df_neighbors['control'] = temp_ctr_df
        # clean
        del ind_center_ctr, knn_1, temp_ctr_df, indices_1, distances_1

        # tst neighborhood: update first the search space with tst center)
        temp_search_tst_group = search_tst_group.iloc[[ind] + non_protected_indices].copy()
        temp_search_tst_group.reset_index(inplace=True, )
        temp_search_tst_group.rename(columns={'index': 'org_index'}, inplace=True)
        # tst neighborhood
        knn_2 = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric=d).fit(temp_search_tst_group[feat_rlvt])
        distances_2, indices_2 = knn_2.kneighbors(ind_center_tst)

        temp_tst_df = pd.DataFrame()
        temp_tst_df['knn_indices'] = pd.Series(indices_2[0])
        temp_tst_df['knn_distances'] = pd.Series(distances_2[0])
        temp_tst_df.sort_values(by='knn_distances', ascending=True, inplace=True)
        temp_tst_df = temp_tst_df.merge(temp_search_tst_group[['org_index']], how='inner', left_on='knn_indices', right_index=True)
        temp_tst_df = temp_tst_df.merge(cf_df[feat_list], how='inner', left_on='org_index', right_index=True)
        # store
        temp_dict_df_neighbors['test'] = temp_tst_df
        # clean
        del ind_center_tst, knn_2, temp_tst_df, indices_2, distances_2, temp_search_tst_group

        # store neighbors for ind
        dict_df_neighbors[int(ind)] = temp_dict_df_neighbors
        del temp_dict_df_neighbors

    return dict_df_neighbors


def get_wald_ci(dict_df_neighbors: Dict[int, Dict[str, DataFrame]],
                feat_trgt: str, feat_trgt_vals: Dict,
                alpha: float = 0.05, tau: float = 0.05,
                ) -> DataFrame:

    # output
    wald_ci = []

    print(f"using significance level of {100*alpha}%")
    z_score = round(st.norm.ppf(1 - (alpha / 2)), 2)

    for ind in dict_df_neighbors:

        ctr_group = dict_df_neighbors[ind]['control']
        # ctr_group = ctr_group.merge(df[feat_list], how='inner', left_on='org_index', right_index=True)

        tst_group = dict_df_neighbors[ind]['test']
        # tst_group = tst_group.merge(cf_df[feat_list], how='inner', left_on='org_index', right_index=True)

        p1 = ctr_group[ctr_group[feat_trgt] == feat_trgt_vals['neg']].shape[0] / ctr_group.shape[0]
        p2 = tst_group[tst_group[feat_trgt] == feat_trgt_vals['neg']].shape[0] / tst_group.shape[0]
        k1 = ctr_group.shape[0]
        k2 = tst_group.shape[0]

        d_alpha = z_score * math.sqrt((p1 * (1 - p1) / k1) + (p2 * (1 - p2) / k2))
        conf_inter = [(p1 - p2) - d_alpha, (p1 - p2) + d_alpha]
        org_diff = round(p1 - p2, 3)
        if (p1 - p2) >= 0:  # from ST paper #1 todo: can also do some tests on the attributes?
            diff = max(0, p1 - p2 - d_alpha)
        else:
            diff = min(0, p1 - p2 + d_alpha)

        # discrimination evidence:
        if org_diff > tau:  # from ST paper #2 todo: can also use CIs?
            cf_st = 'Yes'
        else:
            cf_st = 'No'

        # per ind results
        wald_ci.append(
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

        # clean
        del ctr_group, tst_group, p1, p2, k1, k2, org_diff, d_alpha, conf_inter, diff, cf_st

    # store results as a df
    df = pd.DataFrame(wald_ci)

    return df

#
# EOF
#

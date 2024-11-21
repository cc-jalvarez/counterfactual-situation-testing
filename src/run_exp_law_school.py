import os
import pandas as pd
import numpy as np
from src.situation_testing.situation_testing import SituationTesting

# working directory
wd = os.path.dirname(os.path.dirname(__file__))
# relevant folders
data_path = os.path.abspath(os.path.join(wd, 'data'))
resu_path = os.path.abspath(os.path.join(wd, 'results'))

# load and modify factual data
org_df = pd.read_csv(data_path + '\\clean_LawSchool.csv', sep='|').reset_index(drop=True)
# we focus on sex and race_nonwhite
df = org_df[['sex', 'race_nonwhite', 'LSAT', 'UGPA']].copy()
df.rename(columns={'sex': 'Gender', 'race_nonwhite': 'Race'}, inplace=True)

# the decision maker:
b1 = 0.6
b2 = 0.4
min_score = round(b1*3.93 + b2*46.1, 2)  # 20.8
max_score = round(b1*4.00 + b2*48.00)    # 22
# add the target feature
df['Score'] = b1*df['UGPA'] + b2*df['LSAT']
df['Y'] = np.where(df['Score'] >= min_score, 1, 0)

# k-neighbors
k_list = [15, 30, 50, 100, 250]
# significance level
alpha = 0.05
# tau deviation
tau = 0.0
# type of discrimination
testing_for_negative_disc = True  # TODO

# shared features for all runs
feat_trgt = 'Y'
feat_trgt_vals = {'positive': 1, 'negative': 0}
# list of relevant features
feat_rlvt = ['LSAT', 'UGPA']

########################################################################################################################
# Single discrimination: do(Gender:= Male)
########################################################################################################################

# load and modify counterfactual data
do = 'Male'
org_cf_df = pd.read_csv(data_path + '\\counterfactuals\\' + f'cf_LawSchool_lev3_do{do}.csv', sep='|').reset_index(drop=True)
cf_df = org_cf_df[['Sex', 'Race', 'scf_LSAT', 'scf_UGPA']].copy()
cf_df = cf_df.rename(columns={'Sex': 'Gender', 'scf_LSAT': 'LSAT', 'scf_UGPA': 'UGPA'})

# protected feature
feat_prot = 'Gender'
# values for the protected feature: use 'non_protected' and 'protected'
feat_prot_vals = {'non_protected': 'Male', 'protected': 'Female'}

# add the target feature's counterfactual
cf_df['Score'] = b1*cf_df['UGPA'] + b2*cf_df['LSAT']
cf_df['Y'] = np.where(cf_df['Score'] >= min_score, 1, 0)

# for the loop
test_df = df.copy()
test_cfdf = cf_df.copy()
n_pro = df[df['Gender'] == 'Female'].shape[0]
k_res_abs = []
k_res_prc = []
sigfig = 2

for k in k_list:
    print(k)

    # --- Situation Testing (ST)
    st = SituationTesting()
    st.setup_baseline(test_df, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    st.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, k=k, alpha=alpha, tau=tau)
    st_td = st.get_test_discrimination()
    del st

    # --- Counterfactual Situation Testing without centers (CST wo)
    cst_wo = SituationTesting()
    cst_wo.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    cst_wo.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=False, k=k, alpha=alpha, tau=tau)
    cst_wo_td = cst_wo.get_test_discrimination()
    del cst_wo

    # --- Counterfactual Situation Testing with centers (CST wi)
    cst_wi = SituationTesting()
    cst_wi.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    cst_wi.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=True, k=k, alpha=alpha, tau=tau)
    cst_wi_td = cst_wi.get_test_discrimination()
    # Includes Counterfactual Fairness (CF)
    cf = cst_wi.res_counterfactual_unfairness
    del cst_wi

    # --- k's results: absolutes
    k_res_abs.append(
        {
            'k': k,
            # Num. of discrimination cases
            'ST': st_td[st_td['DiscEvi'] == 'Yes'].shape[0],
            'CSTwo': cst_wo_td[cst_wo_td['DiscEvi'] == 'Yes'].shape[0],
            'CSTwi': cst_wi_td[cst_wi_td['DiscEvi'] == 'Yes'].shape[0],
            'CF': sum(cf == 1),
            # Num. of discrimination cases that are statistically significant
            'ST_sig': st_td[(st_td['DiscEvi'] == 'Yes') & (st_td['StatEvi'] == 'Yes')].shape[0],
            'CSTwo_sig': cst_wo_td[(cst_wo_td['DiscEvi'] == 'Yes') & (cst_wo_td['StatEvi'] == 'Yes')].shape[0],
            'CSTwi_sig': cst_wi_td[(cst_wi_td['DiscEvi'] == 'Yes') & (cst_wi_td['StatEvi'] == 'Yes')].shape[0],
            'CF_sig': cst_wi_td[cst_wi_td['individual'].isin(cf[cf == 1].index.to_list()) & (cst_wi_td['StatEvi'] == 'Yes')].shape[0]
        }
    )

    # --- k's results: percentages
    k_res_prc.append(
        {
            'k': k,
            # % of discrimination cases
            'ST': round(k_res_abs[-1]['ST'] / n_pro * 100, sigfig),
            'CSTwo': round(k_res_abs[-1]['CSTwo'] / n_pro * 100, sigfig),
            'CSTwi': round(k_res_abs[-1]['CSTwi'] / n_pro * 100, sigfig),
            'CF': round(k_res_abs[-1]['CF'] / n_pro * 100, sigfig),
            # % of discrimination cases that are statistically significant
            'ST_sig': round(k_res_abs[-1]['ST_sig'] / n_pro * 100, sigfig),
            'CSTwo_sig': round(k_res_abs[-1]['CSTwo_sig'] / n_pro * 100, sigfig),
            'CSTwi_sig': round(k_res_abs[-1]['CSTwi_sig'] / n_pro * 100, sigfig),
            'CF_sig': round(k_res_abs[-1]['CF_sig'] / n_pro * 100, sigfig)
        }
    )

    del st_td, cst_wo_td, cst_wi_td, cf

print('===== DONE =====')

df_k_res_abs = pd.DataFrame(k_res_abs)
del k_res_abs
df_k_res_prc = pd.DataFrame(k_res_prc)
del k_res_prc

print(df_k_res_abs)
print(df_k_res_prc)

df_k_res_abs.to_csv(resu_path + f'\\res_{do}_LawSchool.csv', sep='|', index=False)
df_k_res_prc.to_csv(resu_path + f'\\res_{do}_LawSchool.csv', sep='|', index=False, mode='a')
del df_k_res_abs, df_k_res_prc

########################################################################################################################
# Single discrimination: do(Race:= White)
########################################################################################################################

# load and modify counterfactual data
do = 'White'
org_cf_df = pd.read_csv(data_path + '\\counterfactuals\\' + f'cf_LawSchool_lev3_do{do}.csv', sep='|').reset_index(drop=True)
cf_df = org_cf_df[['Sex', 'Race', 'scf_LSAT', 'scf_UGPA']].copy()
cf_df = cf_df.rename(columns={'Sex': 'Gender', 'scf_LSAT': 'LSAT', 'scf_UGPA': 'UGPA'})

# add the decision maker
cf_df['Score'] = b1*cf_df['UGPA'] + b2*cf_df['LSAT']
cf_df['Y'] = np.where(cf_df['Score'] >= min_score, 1, 0)

# protected feature
feat_prot = 'Race'
# values for the protected feature: use 'non_protected' and 'protected' accordingly
feat_prot_vals = {'non_protected': 'White', 'protected': 'NonWhite'}

# for the loop
test_df = df.copy()
test_cfdf = cf_df.copy()
n_pro = df[df['Race'] == 'NonWhite'].shape[0]
k_res_abs = []
k_res_prc = []
sigfig = 2

for k in k_list:
    print(k)

    # --- Situation Testing (ST)
    st = SituationTesting()
    st.setup_baseline(test_df, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    st.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, k=k, alpha=alpha, tau=tau)
    st_td = st.get_test_discrimination()
    del st

    # --- Counterfactual Situation Testing without centers (CST wo)
    cst_wo = SituationTesting()
    cst_wo.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    cst_wo.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=False, k=k, alpha=alpha, tau=tau)
    cst_wo_td = cst_wo.get_test_discrimination()
    del cst_wo

    # --- Counterfactual Situation Testing with centers (CST wi)
    cst_wi = SituationTesting()
    cst_wi.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    cst_wi.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=True, k=k, alpha=alpha, tau=tau)
    cst_wi_td = cst_wi.get_test_discrimination()
    # Includes Counterfactual Fairness (CF)
    cf = cst_wi.res_counterfactual_unfairness
    del cst_wi

    # --- k's results: absolutes
    k_res_abs.append(
        {
            'k': k,
            # Num. of discrimination cases
            'ST': st_td[st_td['DiscEvi'] == 'Yes'].shape[0],
            'CSTwo': cst_wo_td[cst_wo_td['DiscEvi'] == 'Yes'].shape[0],
            'CSTwi': cst_wi_td[cst_wi_td['DiscEvi'] == 'Yes'].shape[0],
            'CF': sum(cf == 1),
            # Num. of discrimination cases that are statistically significant
            'ST_sig': st_td[(st_td['DiscEvi'] == 'Yes') & (st_td['StatEvi'] == 'Yes')].shape[0],
            'CSTwo_sig': cst_wo_td[(cst_wo_td['DiscEvi'] == 'Yes') & (cst_wo_td['StatEvi'] == 'Yes')].shape[0],
            'CSTwi_sig': cst_wi_td[(cst_wi_td['DiscEvi'] == 'Yes') & (cst_wi_td['StatEvi'] == 'Yes')].shape[0],
            'CF_sig': cst_wi_td[cst_wi_td['individual'].isin(cf[cf == 1].index.to_list()) & (cst_wi_td['StatEvi'] == 'Yes')].shape[0]
        }
    )

    # --- k's results: percentages
    k_res_prc.append(
        {
            'k': k,
            # % of discrimination cases
            'ST': round(k_res_abs[-1]['ST'] / n_pro * 100, sigfig),
            'CSTwo': round(k_res_abs[-1]['CSTwo'] / n_pro * 100, sigfig),
            'CSTwi': round(k_res_abs[-1]['CSTwi'] / n_pro * 100, sigfig),
            'CF': round(k_res_abs[-1]['CF'] / n_pro * 100, sigfig),
            # % of discrimination cases that are statistically significant
            'ST_sig': round(k_res_abs[-1]['ST_sig'] / n_pro * 100, sigfig),
            'CSTwo_sig': round(k_res_abs[-1]['CSTwo_sig'] / n_pro * 100, sigfig),
            'CSTwi_sig': round(k_res_abs[-1]['CSTwi_sig'] / n_pro * 100, sigfig),
            'CF_sig': round(k_res_abs[-1]['CF_sig'] / n_pro * 100, sigfig)
        }
    )

    del st_td, cst_wo_td, cst_wi_td, cf

print('===== DONE =====')

df_k_res_abs = pd.DataFrame(k_res_abs)
del k_res_abs
df_k_res_prc = pd.DataFrame(k_res_prc)
del k_res_prc

print(df_k_res_abs)
print(df_k_res_prc)

df_k_res_abs.to_csv(resu_path + f'\\res_{do}_LawSchool.csv', sep='|', index=False)
df_k_res_prc.to_csv(resu_path + f'\\res_{do}_LawSchool.csv', sep='|', index=False, mode='a')
del df_k_res_abs, df_k_res_prc

########################################################################################################################
# Multiple discrimination: do(Gender:= Female) + do(Race:= White)
########################################################################################################################

# can run for each k... use the lists k_m_res and k_w_res
df_res_multiple_k = pd.DataFrame(index=['stST', 'cfST', 'cfST_w', 'CF'])
df_res_multiple_p = pd.DataFrame(index=['stST', 'cfST', 'cfST_w', 'CF'])
res_multiple_k = dict()
res_multiple_p = dict()

# for percentages of complainants:
n_pro = org_df[(org_df['sex'] == 'Female') & (org_df['race_nonwhite'] == 'NonWhite')].shape[0]

for df_i in range(len(k_list)):

    temp_k = []
    temp_p = []

    # ST
    temp_k.append(pd.merge(left=k_m_res[df_i][k_m_res[df_i]['ST'] > tau],
                           right=k_w_res[df_i][k_w_res[df_i]['ST'] > tau],
                           how='inner', left_index=True, right_index=True).shape[0]
                  )
    temp_p.append(round(temp_k[-1]/n_pro, 2))

    # CST (w/o)
    temp_k.append(pd.merge(left=k_m_res[df_i][k_m_res[df_i]['cfST'] > tau],
                           right=k_w_res[df_i][k_w_res[df_i]['cfST'] > tau],
                           how='inner', left_index=True, right_index=True).shape[0]
                  )
    temp_p.append(round(temp_k[-1] / n_pro, 2))

    # CST (w)
    temp_k.append(pd.merge(left=k_m_res[df_i][k_m_res[df_i]['cfST_w'] > tau],
                           right=k_w_res[df_i][k_w_res[df_i]['cfST_w'] > tau],
                           how='inner', left_index=True, right_index=True).shape[0]
                  )
    temp_p.append(round(temp_k[-1] / n_pro, 2))

    # CF
    temp_k.append(pd.merge(left=k_m_res[df_i][k_m_res[df_i]['CF'] == 1],
                           right=k_w_res[df_i][k_w_res[df_i]['CF'] == 1],
                           how='inner', left_index=True, right_index=True).shape[0]
                  )
    temp_p.append(round(temp_k[-1] / n_pro, 2))

    print(temp_k)
    print(temp_p)

    res_multiple_k[k_list[df_i]] = temp_k
    res_multiple_p[k_list[df_i]] = temp_p

print('DONE')

for k in res_multiple_k.keys():
    df_res_multiple_k[f'k={k}'] = res_multiple_k[k]
print(df_res_multiple_k)

for k in res_multiple_p.keys():
    df_res_multiple_p[f'k={k}'] = res_multiple_p[k]
print(df_res_multiple_p)

df_res_multiple_k.to_csv(resu_path + f'\\res_Multiple_LawSchool.csv', sep='|', index=True)
df_res_multiple_p.to_csv(resu_path + f'\\res_Multiple_LawSchool.csv', sep='|', index=True, mode='a')

########################################################################################################################
# Intersectional discrimination: do(Gender:= Female) & do(Race:= White)
########################################################################################################################

do = 'MaleWhite'
org_cf_df = \
    pd.read_csv(data_path + '\\counterfactuals\\' + f'cf_LawSchool_lev3_do{do}.csv', sep='|').reset_index(drop=True)

cf_df = org_cf_df[['GenderRace', 'scf_LSAT', 'scf_UGPA']].copy()
cf_df = cf_df.rename(columns={'scf_LSAT': 'LSAT', 'scf_UGPA': 'UGPA'})

# add the decision maker
cf_df['Score'] = b1*cf_df['UGPA'] + b2*cf_df['LSAT']
cf_df['Y'] = np.where(cf_df['Score'] >= min_score, 1, 0)

# add the intersectional var to df
df['GenderRace'] = df['Gender'] + '-' + df['Race']

# store do:=White results
int_res_df = df[['Gender', 'Race', 'Y']].copy()
int_res_df['cf_Y'] = cf_df[['Y']].copy()
# store for all k
k_int_res = []

# attribute-specific params
feat_trgt = 'Y'
feat_trgt_vals = {'positive': 1, 'negative': 0}
# list of relevant features
feat_rlvt = ['LSAT', 'UGPA']
# protected feature
feat_prot = 'GenderRace'
# values for the protected feature: use 'non_protected' and 'protected' accordingly
feat_prot_vals = {
    'non_protected': ['Female-White', 'Male-NonWhite', 'Male-NonWhite', 'Male-White'],
    'protected': 'Female-NonWhite'
                 }

# for percentages of complainants:
n_pro = df[df['GenderRace'] == 'Female-NonWhite'].shape[0]

# run experiments
for k in k_list:

    temp_k = []
    temp_p = []
    temp_k_pos = []
    temp_p_pos = []

    # Standard Situation Testing
    test_df = df.copy()
    st = SituationTesting()

    st.setup_baseline(test_df,
                      nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    int_res_df['ST'] = st.run(target_att=feat_trgt, target_val=feat_trgt_vals,
                              sensitive_att=feat_prot, sensitive_val=feat_prot_vals,
                              k=k, alpha=alpha, tau=tau)

    temp_k.append(int_res_df[int_res_df['ST'] > tau].shape[0])
    temp_p.append(round(int_res_df[int_res_df['ST'] > tau].shape[0] / n_pro * 100, 2))
    temp_k_pos.append(int_res_df[int_res_df['ST'] < tau].shape[0])
    temp_p_pos.append(round(int_res_df[int_res_df['ST'] < tau].shape[0] / n_pro * 100, 2))

    # Counterfactual Situation Testing
    test_df = df.copy()
    test_cfdf = cf_df.copy()
    cf_st = SituationTesting()

    cf_st.setup_baseline(test_df, test_cfdf,
                         nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    int_res_df['cfST'] = cf_st.run(target_att=feat_trgt, target_val=feat_trgt_vals,
                                   sensitive_att=feat_prot, sensitive_val=feat_prot_vals,
                                   include_centers=False,
                                   k=k, alpha=alpha, tau=tau)

    temp_k.append(int_res_df[int_res_df['cfST'] > tau].shape[0])
    temp_p.append(round(int_res_df[int_res_df['cfST'] > tau].shape[0] / n_pro * 100, 2))
    temp_k_pos.append(int_res_df[int_res_df['cfST'] < tau].shape[0])
    temp_p_pos.append(round(int_res_df[int_res_df['cfST'] < tau].shape[0] / n_pro * 100, 2))

    # Counterfactual Situation Testing (including ctr and tst centers)
    test_df = df.copy()
    test_cfdf = cf_df.copy()
    cf_st = SituationTesting()

    cf_st.setup_baseline(test_df, test_cfdf,
                         nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    int_res_df['cfST_w'] = cf_st.run(target_att=feat_trgt, target_val=feat_trgt_vals,
                                     sensitive_att=feat_prot, sensitive_val=feat_prot_vals,
                                     include_centers=True,
                                     k=k, alpha=alpha, tau=tau)

    temp_k.append(int_res_df[int_res_df['cfST_w'] > tau].shape[0])
    temp_p.append(round(int_res_df[int_res_df['cfST_w'] > tau].shape[0] / n_pro * 100, 2))
    temp_k_pos.append(int_res_df[int_res_df['cfST_w'] < tau].shape[0])
    temp_p_pos.append(round(int_res_df[int_res_df['cfST_w'] < tau].shape[0] / n_pro * 100, 2))

    # Counterfactual Fairness
    int_res_df['CF'] = cf_st.res_counterfactual_unfairness

    temp_k.append(int_res_df[int_res_df['CF'] == 1].shape[0])
    temp_p.append(round(int_res_df[int_res_df['CF'] == 1].shape[0] / n_pro * 100, 2))
    temp_k_pos.append(int_res_df[int_res_df['CF'] == 2].shape[0])
    temp_p_pos.append(round(int_res_df[int_res_df['CF'] == 2].shape[0] / n_pro * 100, 2))
    del test_df

    dic_res_k[k] = temp_k
    dic_res_p[k] = temp_p
    dic_res_k_pos[k] = temp_k_pos
    dic_res_p_pos[k] = temp_p_pos

    k_int_res.append(int_res_df)
print('DONE')

for k in dic_res_k.keys():
    res_k[f'k={k}'] = dic_res_k[k]
print(res_k)

for k in dic_res_p.keys():
    res_p[f'k={k}'] = dic_res_p[k]
print(res_p)

res_k.to_csv(resu_path + f'\\res_{do}_LawSchool.csv', sep='|', index=True)
res_p.to_csv(resu_path + f'\\res_{do}_LawSchool.csv', sep='|', index=True, mode='a')

for k in dic_res_k_pos.keys():
    res_k_pos[f'k={k}'] = dic_res_k_pos[k]
print(res_k_pos)

for k in dic_res_p_pos.keys():
    res_p_pos[f'k={k}'] = dic_res_p_pos[k]
print(res_p_pos)

res_k_pos.to_csv(resu_path + f'\\res_pos_{do}_LawSchool.csv', sep='|', index=True)
res_p_pos.to_csv(resu_path + f'\\res_pos_{do}_LawSchool.csv', sep='|', index=True, mode='a')

#
# EOF
#

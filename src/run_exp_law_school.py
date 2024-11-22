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
k_list = [15]  # k_list = [15, 30, 50, 100, 250]
# significance level
alpha = 0.05
# tau deviation
tau = 0.0
# type of discrimination
testing_for_negative_disc = True  # TODO

# shared features/params for all runs
feat_trgt = 'Y'
feat_trgt_vals = {'positive': 1, 'negative': 0}
feat_rlvt = ['LSAT', 'UGPA']
sigfig = 2

print('###############################################################################################################')
print('# Single discrimination: do(Gender:= Male)')
print('###############################################################################################################')

# load and modify counterfactual data
do = 'Male'
org_cf_df = pd.read_csv(data_path + '\\counterfactuals\\' + f'cf_LawSchool_lev3_do{do}.csv', sep='|').reset_index(drop=True)
cf_df = org_cf_df[['Sex', 'Race', 'scf_LSAT', 'scf_UGPA']].copy()
cf_df = cf_df.rename(columns={'Sex': 'Gender', 'scf_LSAT': 'LSAT', 'scf_UGPA': 'UGPA'})

# protected feature
feat_prot = 'Gender'
# values for the protected feature: use 'non_protected' and 'protected'
feat_prot_vals = {'non_protected': 'Male',
                  'protected': 'Female'}

# add the target feature's counterfactual
cf_df['Score'] = b1*cf_df['UGPA'] + b2*cf_df['LSAT']
cf_df['Y'] = np.where(cf_df['Score'] >= min_score, 1, 0)

# for the loop
test_df = df.copy()
test_cfdf = cf_df.copy()
n_pro = df[df['Gender'] == 'Female'].shape[0]
k_res_abs = []
k_res_prc = []

# store results for multiple disc.: G for gender
g_k_res = {}

for k in k_list:
    print(k)
    g_k_res[k] = {}

    # --- Situation Testing (ST)
    st = SituationTesting()
    st.setup_baseline(test_df, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    st.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, k=k, alpha=alpha, tau=tau)
    st_td = st.get_test_discrimination()
    del st
    g_k_res[k]['ST'] = st_td

    # --- Counterfactual Situation Testing without centers (CST wo)
    cst_wo = SituationTesting()
    cst_wo.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    cst_wo.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=False, k=k, alpha=alpha, tau=tau)
    cst_wo_td = cst_wo.get_test_discrimination()
    del cst_wo
    g_k_res[k]['CSTwo'] = cst_wo_td

    # --- Counterfactual Situation Testing with centers (CST wi)
    cst_wi = SituationTesting()
    cst_wi.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    cst_wi.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=True, k=k, alpha=alpha, tau=tau)
    cst_wi_td = cst_wi.get_test_discrimination()
    # Includes Counterfactual Fairness (CF)
    cf = cst_wi.res_counterfactual_unfairness
    del cst_wi
    g_k_res[k]['CSTwi'] = cst_wi_td
    g_k_res[k]['CF'] = cf

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

print('###############################################################################################################')
print('# Single discrimination: do(Race:= White)')
print('###############################################################################################################')

# load and modify counterfactual data
do = 'White'
org_cf_df = pd.read_csv(data_path + '\\counterfactuals\\' + f'cf_LawSchool_lev3_do{do}.csv', sep='|').reset_index(drop=True)
cf_df = org_cf_df[['Sex', 'Race', 'scf_LSAT', 'scf_UGPA']].copy()
cf_df = cf_df.rename(columns={'Sex': 'Gender', 'scf_LSAT': 'LSAT', 'scf_UGPA': 'UGPA'})

# add the target feature's counterfactual
cf_df['Score'] = b1*cf_df['UGPA'] + b2*cf_df['LSAT']
cf_df['Y'] = np.where(cf_df['Score'] >= min_score, 1, 0)

# protected feature
feat_prot = 'Race'
# values for the protected feature: use 'non_protected' and 'protected' accordingly
feat_prot_vals = {'non_protected': 'White',
                  'protected': 'NonWhite'}

# for the loop
test_df = df.copy()
test_cfdf = cf_df.copy()
n_pro = df[df['Race'] == 'NonWhite'].shape[0]
k_res_abs = []
k_res_prc = []

# store results for multiple disc.: R for race
r_k_res = {}

for k in k_list:
    print(k)
    r_k_res[k] = {}

    # --- Situation Testing (ST)
    st = SituationTesting()
    st.setup_baseline(test_df, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    st.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, k=k, alpha=alpha, tau=tau)
    st_td = st.get_test_discrimination()
    del st
    r_k_res[k]['ST'] = st_td

    # --- Counterfactual Situation Testing without centers (CST wo)
    cst_wo = SituationTesting()
    cst_wo.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    cst_wo.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=False, k=k, alpha=alpha, tau=tau)
    cst_wo_td = cst_wo.get_test_discrimination()
    del cst_wo
    r_k_res[k]['CSTwo'] = cst_wo_td

    # --- Counterfactual Situation Testing with centers (CST wi)
    cst_wi = SituationTesting()
    cst_wi.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['LSAT', 'UGPA'])
    cst_wi.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=True, k=k, alpha=alpha, tau=tau)
    cst_wi_td = cst_wi.get_test_discrimination()
    # Includes Counterfactual Fairness (CF)
    cf = cst_wi.res_counterfactual_unfairness
    del cst_wi
    r_k_res[k]['CSTwi'] = cst_wi_td
    r_k_res[k]['CF'] = cf

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

print('###############################################################################################################')
print('# Multiple discrimination: do(Gender:= Female) + do(Race:= White)')
print('###############################################################################################################')

# for the loop
n_pro = org_df[(org_df['sex'] == 'Female') & (org_df['race_nonwhite'] == 'NonWhite')].shape[0]
k_res_abs = []
k_res_prc = []

for k in k_list:
    print(k)
    # use g_k_res and r_k_res: these are nested dictionaries using k run and method

    # --- k's results: absolutes
    k_res_abs.append(
        {
            'k': k,
            # Num. of discrimination cases
            'ST': sum((g_k_res[k]['ST']['DiscEvi'] == 'Yes') & (r_k_res[k]['ST']['DiscEvi'] == 'Yes')),
            'CSTwo': sum((g_k_res[k]['CSTwo']['DiscEvi'] == 'Yes') & (r_k_res[k]['CSTwo']['DiscEvi'] == 'Yes')),
            'CSTwi': sum((g_k_res[k]['CSTwi']['DiscEvi'] == 'Yes') & (r_k_res[k]['CSTwi']['DiscEvi'] == 'Yes')),
            'CF': sum((g_k_res[k]['CF'] == 1.0) & (r_k_res[k]['CF'] == 1.0)),
            # Num. of discrimination cases that are statistically significant
            'ST_sig': sum(
                ((g_k_res[k]['ST']['DiscEvi'] == 'Yes') & (r_k_res[k]['ST']['StatEvi'] == 'Yes')) &
                ((g_k_res[k]['ST']['DiscEvi'] == 'Yes') & (r_k_res[k]['ST']['StatEvi'] == 'Yes'))
            ),
            'CSTwo_sig': sum(
                ((g_k_res[k]['CSTwo']['DiscEvi'] == 'Yes') & (r_k_res[k]['CSTwo']['StatEvi'] == 'Yes')) &
                ((g_k_res[k]['CSTwo']['DiscEvi'] == 'Yes') & (r_k_res[k]['CSTwo']['StatEvi'] == 'Yes'))),
            'CSTwi_sig': sum(
                ((g_k_res[k]['CSTwi']['DiscEvi'] == 'Yes') & (r_k_res[k]['CSTwi']['StatEvi'] == 'Yes')) &
                ((g_k_res[k]['CSTwi']['DiscEvi'] == 'Yes') & (r_k_res[k]['CSTwi']['StatEvi'] == 'Yes'))
            ),
            'CF_sig': sum(
                (g_k_res[k]['CSTwi']['individual'].isin(g_k_res[k]['CF'][g_k_res[k]['CF'] == 1.0].index.to_list()) &
                    g_k_res[k]['CSTwi']['StatEvi'] == 'Yes')
                &
                (r_k_res[k]['CSTwi']['individual'].isin(r_k_res[k]['CF'][r_k_res[k]['CF'] == 1.0].index.to_list()) &
                    r_k_res[k]['CSTwi']['StatEvi'] == 'Yes')
            ),
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

print('===== DONE =====')
del g_k_res, r_k_res

df_k_res_abs = pd.DataFrame(k_res_abs)
del k_res_abs
df_k_res_prc = pd.DataFrame(k_res_prc)
del k_res_prc

print(df_k_res_abs)
print(df_k_res_prc)

df_k_res_abs.to_csv(resu_path + f'\\res_Multiple_LawSchool.csv', sep='|', index=False)
df_k_res_prc.to_csv(resu_path + f'\\res_Multiple_LawSchool.csv', sep='|', index=False, mode='a')
del df_k_res_abs, df_k_res_prc

print('###############################################################################################################')
print('# Intersectional discrimination: do(Gender:= Female) & do(Race:= White)')
print('###############################################################################################################')

# load and modify counterfactual data
do = 'MaleWhite'
org_cf_df = pd.read_csv(data_path + '\\counterfactuals\\' + f'cf_LawSchool_lev3_do{do}.csv', sep='|').reset_index(drop=True)
cf_df = org_cf_df[['GenderRace', 'scf_LSAT', 'scf_UGPA']].copy()
cf_df = cf_df.rename(columns={'scf_LSAT': 'LSAT', 'scf_UGPA': 'UGPA'})

# add the target feature's counterfactual
cf_df['Score'] = b1*cf_df['UGPA'] + b2*cf_df['LSAT']
cf_df['Y'] = np.where(cf_df['Score'] >= min_score, 1, 0)

# Note: add the intersectional feature to df
df['GenderRace'] = df['Gender'] + '-' + df['Race']

# protected feature
feat_prot = 'GenderRace'
# values for the protected feature: use 'non_protected' and 'protected'
feat_prot_vals = {'non_protected': ['Female-White', 'Male-NonWhite', 'Male-NonWhite', 'Male-White'],
                  'protected': 'Female-NonWhite'}

# for the loop
test_df = df.copy()
test_cfdf = cf_df.copy()
n_pro = df[df['GenderRace'] == 'Female-NonWhite'].shape[0]
k_res_abs = []
k_res_prc = []

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

#
# EOF
#

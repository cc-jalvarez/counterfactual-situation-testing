import os
import pandas as pd
import numpy as np
from src.situation_testing.situation_testing import SituationTesting

# working directory
wd = os.path.dirname(os.path.dirname(__file__))
# relevant folders
data_path = os.path.abspath(os.path.join(wd, 'data'))
resu_path = os.path.abspath(os.path.join(wd, 'results'))

# --- load data
df = pd.read_csv(data_path + '\\LoanApplication_v2.csv', sep='|')
cf_df = pd.read_csv(data_path + '\\counterfactuals\\cf_LoanApplication_v2.csv', sep='|')

# --- situation testing parameters
feat_trgt = 'LoanApproval'
# values for the target feature: use 'positive' and 'negative' accordingly
feat_trgt_vals = {'positive': 1, 'negative': -1}
# list of relevant features
feat_rlvt = ['AnnualSalary', 'AccountBalance']
# protected feature
feat_prot = 'Gender'
# values for the protected feature: use 'non_protected' and 'protected' accordingly
feat_prot_vals = {'non_protected': 0, 'protected': 1}

# k-neighbors
k_list = [15, 30, 50, 100, 250]
# significance level
alpha = 0.05
# tau deviation
tau = 0.0

# for percentages of complainants:
n_pro = df[df['Gender'] == 1].shape[0]

# discrimination
res_k = pd.DataFrame(index=['stST', 'cfST', 'cfST_w', 'CF'])
dic_res_k = {}
res_p = pd.DataFrame(index=['stST', 'cfST', 'cfST_w', 'CF'])
dic_res_p = {}
# track statistical significance
res_k_sig = pd.DataFrame(index=['stST', 'cfST', 'cfST_w', 'CF'])
dic_res_k_sig = {}
res_p_sig = pd.DataFrame(index=['stST', 'cfST', 'cfST_w', 'CF'])
dic_res_p_sig = {}

# positive discrimination
res_k_pos = pd.DataFrame(index=['stST', 'cfST', 'cfST_w', 'CF'])
dic_res_k_pos = {}
res_p_pos = pd.DataFrame(index=['stST', 'cfST', 'cfST_w', 'CF'])
dic_res_p_pos = {}

for k in k_list:

    temp_k = []
    temp_p = []
    temp_k_sig = []
    temp_p_sig = []
    temp_k_pos = []
    temp_p_pos = []

    # Standard Situation Testing
    test_df = df.copy()
    st = SituationTesting()

    st.setup_baseline(test_df,
                      nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
    test_df['ST'] = st.run(target_att=feat_trgt, target_val=feat_trgt_vals,
                           sensitive_att=feat_prot, sensitive_val=feat_prot_vals,
                           k=k, alpha=alpha, tau=tau)
    # discrimination
    temp_k.append(test_df[test_df['ST'] > tau].shape[0])
    temp_p.append(round(test_df[test_df['ST'] > tau].shape[0] / n_pro * 100, 2))
    # discrimination: statistically significant
    temp_test_disc = st.get_test_discrimination()
    star_diff = temp_test_disc[(temp_test_disc['DiscEvi'] == 'Yes') & (temp_test_disc['StatEvi'] == 'Yes')].shape[0]
    temp_k_sig.append(star_diff)
    temp_p_sig.append(round(star_diff / n_pro * 100, 2))
    del temp_test_disc, star_diff
    # positive discrimination
    temp_k_pos.append(test_df[test_df['ST'] < tau].shape[0])
    temp_p_pos.append(round(test_df[test_df['ST'] < tau].shape[0] / n_pro * 100, 2))
    del test_df

    # Counterfactual Situation Testing
    test_df = df.copy()
    test_cfdf = cf_df.copy()
    cf_st = SituationTesting()

    cf_st.setup_baseline(test_df, test_cfdf,
                         nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
    test_df['cfST'] = cf_st.run(target_att=feat_trgt, target_val=feat_trgt_vals,
                                sensitive_att=feat_prot, sensitive_val=feat_prot_vals,
                                include_centers=False,
                                k=k, alpha=alpha, tau=tau)
    # discrimination
    temp_k.append(test_df[test_df['cfST'] > tau].shape[0])
    temp_p.append(round(test_df[test_df['cfST'] > tau].shape[0] / n_pro * 100, 2))
    # discrimination: statistically significant
    temp_test_disc = cf_st.get_test_discrimination()
    star_diff = temp_test_disc[(temp_test_disc['DiscEvi'] == 'Yes') & (temp_test_disc['StatEvi'] == 'Yes')].shape[0]
    temp_k_sig.append(star_diff)
    temp_p_sig.append(round(star_diff / n_pro * 100, 2))
    del temp_test_disc, star_diff
    # positive discrimination
    temp_k_pos.append(test_df[test_df['cfST'] < tau].shape[0])
    temp_p_pos.append(round(test_df[test_df['cfST'] < tau].shape[0] / n_pro * 100, 2))
    del test_df

    # Counterfactual Situation Testing (including ctr and tst centers)
    test_df = df.copy()
    test_cfdf = cf_df.copy()
    cf_st = SituationTesting()

    cf_st.setup_baseline(test_df, test_cfdf,
                         nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
    test_df['cfST'] = cf_st.run(target_att=feat_trgt, target_val=feat_trgt_vals,
                                sensitive_att=feat_prot, sensitive_val=feat_prot_vals,
                                include_centers=True,
                                k=k, alpha=alpha, tau=tau)
    # discrimination
    temp_k.append(test_df[test_df['cfST'] > tau].shape[0])
    temp_p.append(round(test_df[test_df['cfST'] > tau].shape[0] / n_pro * 100, 2))
    # discrimination: statistically significant
    temp_test_disc = cf_st.get_test_discrimination()
    star_diff = temp_test_disc[(temp_test_disc['DiscEvi'] == 'Yes') & (temp_test_disc['StatEvi'] == 'Yes')].shape[0]
    temp_k_sig.append(star_diff)
    temp_p_sig.append(round(star_diff / n_pro * 100, 2))
    del temp_test_disc, star_diff
    # positive discrimination
    temp_k_pos.append(test_df[test_df['cfST'] < tau].shape[0])
    temp_p_pos.append(round(test_df[test_df['cfST'] < tau].shape[0] / n_pro * 100, 2))

    # Counterfactual Fairness
    test_df['CF'] = cf_st.res_counterfactual_unfairness
    # CF discrimination
    temp_k.append(test_df[test_df['CF'] == 1].shape[0])
    temp_p.append(round(test_df[test_df['CF'] == 1].shape[0] / n_pro * 100, 2))
    # CF discrimination: statistically significant
    temp_test_disc = cf_st.get_test_discrimination()
    star_cf = temp_test_disc[temp_test_disc['individual'].isin(test_df[test_df['CF'] == 1].index.to_list()) & (temp_test_disc['StatEvi'] == 'Yes')].shape[0]
    temp_k_sig.append(star_cf)
    temp_p_sig.append(round(star_cf / n_pro * 100, 2))
    del temp_test_disc, star_cf
    # CF positive discrimination
    temp_k_pos.append(test_df[test_df['CF'] == 2].shape[0])
    temp_p_pos.append(round(test_df[test_df['CF'] == 2].shape[0] / n_pro * 100, 2))
    del test_df

    dic_res_k[k] = temp_k
    dic_res_p[k] = temp_p
    dic_res_k_sig[k] = temp_k_sig
    dic_res_p_sig[k] = temp_p_sig
    dic_res_k_pos[k] = temp_k_pos
    dic_res_p_pos[k] = temp_p_pos

print('===== DONE =====')

print("===> discrimination")
for k in dic_res_k.keys():
    res_k[f'k={k}'] = dic_res_k[k]
print(res_k)
for k in dic_res_p.keys():
    res_p[f'k={k}'] = dic_res_p[k]
print(res_p)

print("===> discrimination: statistically significant")
for k in dic_res_k_sig.keys():
    res_k_sig[f'k={k}'] = dic_res_k_sig[k]
print(res_k_sig)
for k in dic_res_p_sig.keys():
    res_p_sig[f'k={k}'] = dic_res_p_sig[k]
print(res_p_sig)

res_k.to_csv(resu_path + '\\res_LoanApplication.csv', sep='|', index=True)
res_p.to_csv(resu_path + '\\res_LoanApplication.csv', sep='|', index=True, mode='a')
res_k_sig.to_csv(resu_path + '\\res_LoanApplication.csv', sep='|', index=True, mode='a')
res_p_sig.to_csv(resu_path + '\\res_LoanApplication.csv', sep='|', index=True, mode='a')

print("===> positive discrimination")
for k in dic_res_k_pos.keys():
    res_k_pos[f'k={k}'] = dic_res_k_pos[k]
print(res_k_pos)
for k in dic_res_p_pos.keys():
    res_p_pos[f'k={k}'] = dic_res_p_pos[k]
print(res_p_pos)

res_k_pos.to_csv(resu_path + '\\res_pos_LoanApplication.csv', sep='|', index=True)
res_p_pos.to_csv(resu_path + '\\res_pos_LoanApplication.csv', sep='|', index=True, mode='a')

#
# EOF
#

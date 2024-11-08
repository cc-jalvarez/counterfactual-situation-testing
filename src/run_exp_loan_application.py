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
n_pro = df[df[feat_prot] == 1].shape[0]

# for the loop
test_df = df.copy()
test_cfdf = cf_df.copy()
k_res_abs = []
k_res_prc = []
sigfig = 2

for k in k_list:
    print(k)

    # --- Situation Testing (ST)
    st = SituationTesting()
    st.setup_baseline(test_df, nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
    st.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, k=k, alpha=alpha, tau=tau)
    st_td = st.get_test_discrimination()
    del st

    # --- Counterfactual Situation Testing without centers (CST wo)
    cst_wo = SituationTesting()
    cst_wo.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
    cst_wo.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=False, k=k, alpha=alpha, tau=tau)
    cst_wo_td = cst_wo.get_test_discrimination()
    del cst_wo

    # --- Counterfactual Situation Testing with centers (CST wi)
    cst_wi = SituationTesting()
    cst_wi.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
    cst_wi.run(target_att=feat_trgt, target_val=feat_trgt_vals, sensitive_att=feat_prot, sensitive_val=feat_prot_vals, include_centers=True, k=k, alpha=alpha, tau=tau)
    cst_wi_td = cst_wi.get_test_discrimination()
    # it also includes Counterfactual Fairness (CF)
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

df_k_res_abs.to_csv(resu_path + '\\res_LoanApplication.csv', sep='|', index=True)
df_k_res_prc.to_csv(resu_path + '\\res_LoanApplication.csv', sep='|', index=True, mode='a')

#
# EOF
#

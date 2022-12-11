import os
import pandas as pd
import numpy as np
from src.situation_testing.situation_testing import SituationTesting

proj_path = r'C:\\Users\\Jose Alvarez\\Documents\\Projects\\CounterfactualSituationTesting\\'  # delete later (jupyter)
data_path = os.path.abspath(os.path.join(proj_path, 'data'))
resu_path = os.path.abspath(os.path.join(proj_path, 'results', 'counterfactuals'))

# --- load data
df = pd.read_csv(data_path + '\\Karimi2020_v2.csv', sep='|', )
cf_df = pd.read_csv(resu_path + '\\cf_Karimi2020_v2.csv', sep='|', )

# --- Situation testing params
feat_trgt = 'LoanApproval'
feat_trgt_vals = {'positive': 1, 'negative': -1}
# list of relevant features
feat_rlvt = ['AnnualSalary', 'AccountBalance']
# protected feature
feat_prot = 'Gender'
# values for the protected feature: use 'non_protected' and 'protected' accordingly
feat_prot_vals = {'non_protected': 0, 'protected': 1}
# k-neighbors
n = 15
# significance level
alpha = 0.05
# tau deviation
tau = 0.0

# # --- ST
# test_df = df.copy()
#
# st = SituationTesting()
# st.setup_baseline(test_df, nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
#
# test_df['ST'] = st.run(target_att='LoanApproval', target_val={'positive': 1, 'negative': -1},
#                        sensitive_att='Gender', sensitive_val={'non_protected': 0, 'protected': 1},
#                        k=n, alpha=alpha, tau=tau)
# print(test_df[test_df['ST'] > tau].shape[0])
#
# del test_df
#
# # --- cfST without centers
# test_df = df.copy()
# test_cfdf = cf_df.copy()
#
# # don't include the centers
# cf_st = SituationTesting()
# cf_st.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
#
# test_df['cfST'] = cf_st.run(target_att='LoanApproval', target_val={'positive': 1, 'negative': -1},
#                             sensitive_att='Gender', sensitive_val={'non_protected': 0, 'protected': 1},
#                             include_centers=False,
#                             k=n, alpha=alpha, tau=tau)
#
# print(test_df[test_df['cfST'] > tau].shape[0])
#
# del test_df
#
# # --- cfST with centers
# test_df = df.copy()
# test_cfdf = cf_df.copy()
#
# # include the centers
# cf_st = SituationTesting()
# cf_st.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
#
# test_df['cfST'] = cf_st.run(target_att='LoanApproval', target_val={'positive': 1, 'negative': -1},
#                             sensitive_att='Gender', sensitive_val={'non_protected': 0, 'protected': 1},
#                             include_centers=True,
#                             k=n, alpha=alpha, tau=tau)
#
# print(test_df[test_df['cfST'] > tau].shape[0])
#
# del test_df

# For the paper's table:

for new_k in [15, 30, 50, 100]:
    print('===> k={k}'.format(k=new_k))

    print('standard ST')
    test_df = df.copy()

    st = SituationTesting()
    st.setup_baseline(test_df, nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])

    test_df['ST'] = st.run(target_att='LoanApproval', target_val={'positive': 1, 'negative': -1},
                           sensitive_att='Gender', sensitive_val={'non_protected': 0, 'protected': 1},
                           k=new_k, alpha=alpha, tau=tau)
    print(test_df[test_df['ST'] > tau].shape[0])

    del test_df

    print('counterfactual ST (without centers)')
    test_df = df.copy()
    test_cfdf = cf_df.copy()

    cf_st = SituationTesting()
    cf_st.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'],
                         continuous_atts=['AnnualSalary', 'AccountBalance'])

    test_df['cfST'] = cf_st.run(target_att='LoanApproval', target_val={'positive': 1, 'negative': -1},
                                sensitive_att='Gender', sensitive_val={'non_protected': 0, 'protected': 1},
                                include_centers=False,
                                k=new_k, alpha=alpha, tau=tau)

    print(test_df[test_df['cfST'] > tau].shape[0])

    del test_df

    print('counterfactual ST (with centers)')
    test_df = df.copy()
    test_cfdf = cf_df.copy()

    # include the centers
    cf_st = SituationTesting()
    cf_st.setup_baseline(test_df, test_cfdf, nominal_atts=['Gender'],
                         continuous_atts=['AnnualSalary', 'AccountBalance'])

    test_df['cfST'] = cf_st.run(target_att='LoanApproval', target_val={'positive': 1, 'negative': -1},
                                sensitive_att='Gender', sensitive_val={'non_protected': 0, 'protected': 1},
                                include_centers=True,
                                k=new_k, alpha=alpha, tau=tau)

    print(test_df[test_df['cfST'] > tau].shape[0])

    del test_df

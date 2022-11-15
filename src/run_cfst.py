import pandas as pd
import numpy as np
from src.situation_testing.situation_testing import SituationTesting

data_path = r'C:\\Users\\Jose Alvarez\\Documents\\Projects\\CounterfactualSituationTesting\\data\\'
resu_path = r'C:\\Users\\Jose Alvarez\\Documents\\Projects\\CounterfactualSituationTesting\\results\\counterfactuals\\'
df = pd.read_csv(data_path + 'Karimi2020_v2.csv', sep='|', )
cf_df = pd.read_csv(resu_path + 'cf_Karimi2020_v2.csv', sep='|', )
print(df.head(5))
print(cf_df.head(5))

org_df = df.copy()

feat_trgt = 'LoanApproval'
feat_trgt_vals = {'positive': 1, 'negative': -1}
# list of relevant features
feat_rlvt = ['AnnualSalary', 'AccountBalance']
# protected feature
feat_prot = 'Gender'
# values for the protected feature: use 'non_protected' and 'protected' accordingly
feat_prot_vals = {'non_protected': 0, 'protected': 1}

st = SituationTesting()
st.setup_baseline(df=df,
                  cf_df=cf_df,
                  nominal_atts=['Gender'], continuous_atts=['AnnualSalary', 'AccountBalance'])
org_df['diff'] = st.run(target_att='LoanApproval', target_val={'positive': 1, 'negative': -1},
                        sensitive_att='Gender', sensitive_val={'non_protected': 0, 'protected': 1},
                        k=15,
                        include_centers=True)
print(org_df.head(5))




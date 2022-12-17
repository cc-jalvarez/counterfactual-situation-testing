import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# set working directory
wrk_dir = os.path.dirname(os.getcwd())
# set data path
data_path = wrk_dir + '\\' + 'data' + '\\'
# Loan Application v2 (factual) data
org_df = pd.read_csv(data_path + '\\' + 'LoanApplication_v2.csv', sep='|', )

# define the list of features
feat_trgt = ['LoanApproval']
feat_rlvt = ['AnnualSalary', 'AccountBalance']
feat_prot = ['Gender']
do = {'Gender': 0}  # female to male
df = org_df[feat_trgt + feat_rlvt + feat_prot].copy()
print(df.head(5))

# store counterfactual df
cf_df = dict()

# 1) estimate each f in M where needed according to the known causal graph:
# 1.1) create model objects
# f for AnnualSalary
model_sal = LinearRegression(fit_intercept=True, normalize=False)
# f for AccountBalance
model_acc = LinearRegression(fit_intercept=True, normalize=False)

# 1.2) prepare data for the models
# f for Salary
x_sal = np.array(df['Gender'].copy()).reshape((-1, 1))
y_sal = np.array(df['AnnualSalary'].copy())
# f for Account
x_acc = np.array(df[['AnnualSalary', 'Gender']].copy())#.reshape((-1, 1))
y_acc = np.array(df['AccountBalance'].copy())

# 1.3) estimate the models
model_sal.fit(x_sal, y_sal)
model_acc.fit(x_acc, y_acc)

# 2) generate the (structural) counterfactuals (cf) for X using Pearl's abduction, action, prediction steps:
# 2.1) Abduction (or individual error terms given each f)
cf_df['u_AnnualSalary'] = round(df['AnnualSalary'] - model_sal.predict(x_sal), 2)
cf_df['u_AccountBalance'] = round(df['AccountBalance'] - model_acc.predict(x_acc), 2)
cf_df = pd.DataFrame.from_dict(cf_df)

# 2.2) Action + Prediction (X-wise): here, we focus on being female (the protected group)
do_male = np.repeat(0, repeats=df.shape[0]).reshape((-1, 1))
cf_df['AnnualSalary'] = round(model_sal.predict(do_male) + cf_df['u_AnnualSalary'], 2)
do_male2 = cf_df[['AnnualSalary']].copy()
do_male2['Gender'] = do_male
cf_df['AccountBalance'] = round(model_acc.predict(do_male2) + cf_df['u_AccountBalance'], 2)

# 3) Prediction (Y-wise): Generate cf_Y (when b is known)
# b params
beta_0 = 225000
beta_1 = (3/10)
beta_2 = 5
# b functional form
cf_df['LoanApproval'] = np.sign(cf_df['AnnualSalary'] + beta_2*cf_df['AccountBalance'] - beta_0)
# keep track of A (i.e., original gender)
cf_df['Gender'] = df['Gender']
# lose the U's
cf_df = cf_df[feat_trgt + feat_rlvt + feat_prot]

print(cf_df.head(5))

# store counterfactual data in data/counterfactuals
cf_df.to_csv(data_path + '\\counterfactuals\\' + 'cf_LoanApplication_v2.csv', sep='|', index=False)

# Paper figures:
# distributions for paper: X1
b = 100
plt.hist(df[df['Gender'] == 1]['AnnualSalary'], bins=b, alpha=0.9, label=r'$X_1^F$')
plt.hist(cf_df[cf_df['Gender'] == 1]['AnnualSalary'], bins=b, alpha=0.7, label=r'$X_1^{CF}$')
plt.legend(loc='upper right')
plt.ylabel('Frequency')
plt.xlabel(r'Annual salary ($X1$) for females')

# distributions for paper: X2
b = 100
plt.hist(df[df['Gender'] == 1]['AccountBalance'], bins=b, alpha=0.9, label=r'$X_2^F$')
plt.hist(cf_df[cf_df['Gender'] == 1]['AccountBalance'], bins=b, alpha=0.7, label=r'$X_2^{CF}$')
plt.legend(loc='upper right')
plt.ylabel('Frequency')
plt.xlabel(r'Account balance ($X_2$) for females')

#
# EOF
#

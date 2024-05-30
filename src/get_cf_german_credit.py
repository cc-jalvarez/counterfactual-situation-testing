import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# set working directory
wrk_dir = os.path.dirname(os.getcwd())
# set data path
data_path = wrk_dir + '\\' + 'data' + '\\'
# Loan Application v2 (factual) data
org_df = pd.read_csv(data_path + '\\' + 'clean_GermanCreditData.csv', sep='|', )

print(org_df.head())

"""
Consider Fig. 12 from the VACA paper for a loan approval setting... maybe I can come up with my own?
Consider gender, age, education, loan amount, duration, income and savings
Create a decision?
"""


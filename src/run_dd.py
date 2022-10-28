import os
import pandas as pd
import numpy as np
# local
import src.dd as dd
from src.situation_testing import situation_testing as st

# data_path = os.getcwd() + '\\' + 'data' + '\\'
data_path = 'C:\\Users\\Jose Alvarez\\Documents\\Projects\\CounterfactualSituationTesting\\data\\'
# reload factual data (no need to use the counterfactuals)
df = pd.read_csv(data_path + 'Karimi2020_v2.csv', sep='|', )
del df['u1']
del df['u2']
# for 'convinience' make 'Gender' var explicit
df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'})

# nominal-ordinal-continuous partition of predictive attributes (required for distance functions)
nominal_atts = ['Gender']
ordinal_atts = []
continuous_atts = ['AnnualSalary', 'AccountBalance']

# encoding of ordinal attributes as integers plus (optional) encoding of nominal/target attributes
decode = {
    'Gender': {0: 'Male', 1: 'Female'},
    'LoanApproval': {0: -1.0, 1: 1.0}
    }

# predictive attributes (for models)
target = 'LoanApproval'
pred_atts = nominal_atts + ordinal_atts + continuous_atts
print(pred_atts)
pred_all = pred_atts + [target]
print(pred_all)

# protected/unprotected groups attribute
sensitive_att = 'Gender'

# encode nominal, ordinal, and target attribute
df_code = dd.Encode(nominal_atts + ordinal_atts + [target], decode)
df = df_code.fit_transform(df)
# set ordinal type as int (instead of category)
df[ordinal_atts] = df[ordinal_atts].astype(int)

# dictionary of encodings for each discrete column
df_code.decode

# create new df for storing k-nn ST
df2 = df.copy()

# number of neighbors
k = 15

# object for indvidual discrimination analysis
dist = dd.ID(df, nominal_atts, continuous_atts, ordinal_atts)

# protected attribute
pro_att = 'Gender'
unpro_val = df_code.encode['Gender']['Male']

unpro_train = df2[sensitive_att]==unpro_val
pro_train = df2[sensitive_att]!=unpro_val

# sets risk difference for target based on top-k neighbor instances
df2['t'] = dist.topkdiff(df2,              # dataframe
                         unpro_train,      # unprotected
                         pro_train,        # protected (could be a list)
                         target+'=0',      # bad decision
                         dist.kdd2011dist, # distance
                         k)                # k-neighbors
print('done')

# column 't' is p1 - p2, of the diff
print(df2.head(5))





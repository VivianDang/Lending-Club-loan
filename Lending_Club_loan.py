#%%[markdown]
## Overview
# - This notebook includes a case study
# - See all code in [github repository](https://github.com/VivianDang/Lending-Club-loan).
### Description
#%%
from operator import index
import numpy as np
import pandas as pd
from sympy import subsets

#%%
loans = pd.read_csv('loans_full_schema.csv')
df = loans.copy()
col = df.columns
#%%
# useful functions borrowed from Professor Yuxiao Huang
# https://github.com/yuxiaohuang/teaching/blob/master/gwu/machine_learning_I/spring_2022/code/utilities/p2_shallow_learning/pmlm_utilities_shallow.ipynb

# identifier checking
def id_checker(df, dtype='float'):
    """
    The identifier checker

    Parameters
    ----------
    df : dataframe
    dtype : the data type identifiers cannot have, 'float' by default
            i.e., if a feature has this data type, it cannot be an identifier

    Returns
    ----------
    The dataframe of identifiers
    """

    # Get the dataframe of identifiers
    df_id = df[[var for var in df.columns
                # If the data type is not dtype
                if (df[var].dtype != dtype
                    # If the value is unique for each sample
                    and df[var].nunique(dropna=True) == df[var].notnull().sum())]]

    return df_id

# missing values cjecking
def nan_checker(df):
    """
    The NaN checker

    Parameters
    ----------
    df : the dataframe

    Returns
    ----------
    The dataframe of variables with NaN, their proportion of NaN and data type
    """

    # Get the dataframe of variables with NaN, their proportion of NaN and data type
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])

    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)

    return df_nan

# categorical value checking
def cat_var_checker(df, dtype='object'):
    """
    The categorical variable checker

    Parameters
    ----------
    df : the dataframe
    dtype : the data type categorical variables should have, 'object' by default
            i.e., if a variable has this data type, it should be a categorical variable

    Returns
    ----------
    The dataframe of categorical variables and their number of unique value
    """

    # Get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           # If the data type is dtype
                           for var in df.columns if df[var].dtype == dtype],
                          columns=['var', 'nunique'])

    # Sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)

    return df_cat

#%%

# identify ids
print('-'*70)
print('Identifiers Checking: ')
df_id = id_checker(df)
print(df_id.head())
print(f'There are {len(df_id.columns)} identifiers in this dataset.')
print('-'*70)

# identify categorical variables
print('-'*70)
print('Categorical Variables Checking: ')
df_cat = cat_var_checker(df)
print(df_cat)
print(f'There are {len(df_cat)} categorical variables in this dataset.')
print('-'*70)

# identify missing values
print('-'*70)
print('Missing Values Checking: ')
df_nan = nan_checker(df)
print(df_nan)
print(f'There are {len(df_nan)} variables with missing values in this dataset.')
print('-'*70)

#%%[markdown]

## handling missing values  
#
# Since most of the value missings in the joint variables would be due to the two types of applications: Individual and Joint,
# Split the dataset into two groups by application type and check missing values again. 
#%%
# drop missing values with proportion less than 20%
df.dropna(subset=df_nan['var'][-5:], axis=0, inplace=True)
#%%
df_ind = df[df.application_type=='individual'].drop(['verification_income_joint','debt_to_income_joint', 'annual_income_joint'], axis=1)
df_joint = df[df.application_type=='joint']
# %%
# identify missing values
print('-'*70)
print('Missing Values Checking: ')
print('-'*50)
print('Individual Application: ')
df_ind_nan = nan_checker(df_ind)
print(df_ind_nan)
print('There are 6 variables with missing values.')
print('-'*50)
print('Joint Application: ')
df_join_nan = nan_checker(df_joint)
print(df_join_nan)
print('There are 7 variables with missing values.')
print('-'*70)
# %%
# drop missing values with proportion less than 5% in Joint Application
df.drop(index=df_joint.verification_income_joint.isna().index)
df_joint.dropna(subset=['verification_income_joint'], axis=0, inplace=True)
#%%
# fill joint variables in Individual Application with zero
df[['debt_to_income_joint', 'verification_income_joint', 'annual_income_joint']] = df[['debt_to_income_joint', 'verification_income_joint', 'annual_income_joint']].fillna(0)

# check missing values again
print('-'*70)
print('Missing Values Checking: ')
df_nan = nan_checker(df)
print(df_nan)
print(f'There are {len(df_nan)} variables with missing values in this dataset.')
print('-'*70)
#%%
# months_since_last_delinq is comply with delinq_2y
print('-'*70)
print('Missing Values Checking with no Delinquencies: ')
print('-'*50)
print('Individual Application: ')
df_delin = df[df.delinq_2y!=0]
df_delin_nan = nan_checker(df_delin)
print(df_delin_nan)
print('-'*70)
# %%
# fill months_since_last_delinq with zero for those whose delinq_2y=0
df.loc[df.delinq_2y==0, 'months_since_last_delinq'] = 0

# check missing values again
print('-'*70)
print('Missing Values Checking: ')
df_nan = nan_checker(df)
print(df_nan)
print(f'There are {len(df_nan)} variables with missing values in this dataset.')
print('-'*70)
# %%
# since there seems to have no variables comply with monyhs_since_90d_late, we just drop this column
df.drop('months_since_90d_late', inplace=True, axis=1)
print('-'*70)
print('Missing Values Checking: ')
df_nan = nan_checker(df)
print(df_nan)
print(f'There are {len(df_nan)} variables with missing values in this dataset.')
print('-'*70)
#%%
# save clean dataset
df.to_csv('loan_clean.csv', index=False)
# %%[markdown]
## Visualization
#%%

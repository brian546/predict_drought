# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
# %%
df = pd.read_csv('../data/test_timeseries/test_timeseries.csv',parse_dates=['date'])
print(df.info())
# %%
print(df.isnull().sum())

# %%

# number of regions
df['fips'].nunique()

# %%

df.info()

# find out if any multicolinearity

# T2M highly collerated (corr > 0.7) with QV2M,T2MDEW,T2MWET, T2M_MAX, T2M_MIN, and TS 

# WS10M highly collerated  with W10M_MAX, WS10M_MIN, WS50M, WS550M_MAX, WS550M_MIN

# columns left: PRECTOT, PS, T2M, T2M_RANGE, WS10M, WS50M_RANGE

# plt.figure(figsize=(20,20))
# sns.heatmap(df.drop(columns=['fips','date']).corr(),annot=True)

# %%
df['date'].head()

# %%

new_df = df[['fips','date', 'PRECTOT', 'PS', 'T2M', 'T2M_RANGE', 'WS10M', 'WS50M_RANGE','score']]

# %%

# proving that the dought score was released every Tuesday, reviewing the dought condtion in the past week

print(new_df[new_df['date'].dt.weekday == 1]['score'].isnull().sum())

# %%

# filter out row at the date that is after the last Tuesday of 2020 and row at the first Tuesday.

new_df = new_df[(df['date'] > pd.Timestamp(2019,1,1)) & (df['date'] < pd.Timestamp(2020,12,30))]

# %%

# get the mean of data per week 
week_df = df.groupby('fips').resample('W-TUE', on='date').mean()

week_df = week_df.drop(columns=['fips']).reset_index()

week_df['score'] = week_df['score'].round()

# %%
week_df
# %%
week_df['score'].isnull().sum()

# %%
# week_df.to_csv('../data/week_score.csv', index=False)

# plt.figure(figsize=(20,20))

# sns.heatmap(week_df.corr(),annot=True)
# %%

week_df['score'].value_counts()

# %%

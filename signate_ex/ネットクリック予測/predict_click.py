#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


test_data = pd.read_csv('test.tsv',encoding='utf-8',delimiter = '\t')
train_data = pd.read_csv('train.tsv',encoding='utf-8',delimiter = '\t')

import matplotlib.pyplot as plt
import numpy as np

train = train_data
train = pd.DataFrame(train)

import seaborn as sns

#閾値を決めて頻度が少ない値をNaNに置き換える
df_c1 = pd.DataFrame(train['C1'])
threshold = 30000
value_counts = df_c1.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c1.replace(to_remove,np.nan,inplace=True)

pd.get_dummies(df_c1['C1'])


print('C1')


#閾値を決めて頻度が少ない値をNaNに置き換える
df_c2 = pd.DataFrame(train['C2'])
threshold = 90000
value_counts = df_c2.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c2.replace(to_remove,np.nan,inplace=True)

pd.get_dummies(df_c2['C2'])


print('C2')

#閾値を決めて頻度が少ない値をNaNに置き換える
df_c3 = pd.DataFrame(train['C3'])
threshold = 70000
value_counts = df_c3.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c3.replace(to_remove,np.nan,inplace=True)

pd.get_dummies(df_c3['C3'])
print('C3')
#閾値を決めて頻度が少ない値をNaNに置き換える
df_c4 = pd.DataFrame(train['C4'])
threshold = 160000
value_counts = df_c4.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c4.replace(to_remove,np.nan,inplace=True)

pd.get_dummies(df_c4['C4'])

#閾値を決めて頻度が少ない値をNaNに置き換える
df_c5 = pd.DataFrame(train['C5'])
threshold = 110000
value_counts = df_c5.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c5.replace(to_remove,np.nan,inplace=True)

pd.get_dummies(df_c5['C5'])


#閾値を決めて頻度が少ない値をNaNに置き換える
df_c6 = pd.DataFrame(train['C6'])
threshold = 120000
value_counts = df_c6.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c6.replace(to_remove,np.nan,inplace=True)

pd.get_dummies(df_c6['C6'])

#閾値を決めて頻度が少ない値をNaNに置き換える
df_I11 = pd.DataFrame(train['I11'])
threshold = 40000
value_counts =df_I11.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_I11.replace(to_remove,np.nan,inplace=True)

pd.get_dummies(df_I11['I11'])
train_data['I12'].value_counts().head(6)


print('I11')


#閾値を決めて頻度が少ない値をNaNに置き換える
df_I12 = pd.DataFrame(train['I12'])
threshold = 60000
value_counts =df_I12.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_I12.replace(to_remove,np.nan,inplace=True)

pd.get_dummies(df_I12['I12'])
print('I12')

#閾値を決めて頻度が少ない値をNaNに置き換える
df_I13 = pd.DataFrame(train['I13'])
threshold = 500
value_counts =df_I13.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_I13.replace(to_remove,np.nan,inplace=True)
pd.get_dummies(df_I13['I13'])

print('I13')

#閾値を決めて頻度が少ない値をNaNに置き換える
df_I14 = pd.DataFrame(train['I14'])
threshold = 500
value_counts =df_I14.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_I14.replace(to_remove,np.nan,inplace=True)
pd.get_dummies(df_I13['I14'])

# In[ ]:


#データを統合する
df_new = pd.merge([df_c1,df_c2,df_c3,df_c4,df_c5,df_c6,train['I6'],train['I7'],train['I8'],train['I9'],
train['I10'],df_I11,df_I12,df_I13,df_14])

print(df_new.head())


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# In[ ]:

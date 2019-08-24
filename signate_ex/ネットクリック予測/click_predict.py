#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd


# In[103]:




# In[50]:


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
print('C1ダミー化')

#閾値を決めて頻度が少ない値をNaNに置き換える
df_c2 = pd.DataFrame(train['C2'])
threshold = 90000
value_counts = df_c2.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c2.replace(to_remove,np.nan,inplace=True)
print('C2ダミー化')
#閾値を決めて頻度が少ない値をNaNに置き換える
df_c3 = pd.DataFrame(train['C3'])
threshold = 70000
value_counts = df_c3.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c3.replace(to_remove,np.nan,inplace=True)


#閾値を決めて頻度が少ない値をNaNに置き換える
df_c4 = pd.DataFrame(train['C4'])
threshold = 160000
value_counts = df_c4.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c4.replace(to_remove,np.nan,inplace=True)

#閾値を決めて頻度が少ない値をNaNに置き換える
df_c5 = pd.DataFrame(train['C5'])
threshold = 110000
value_counts = df_c5.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c5.replace(to_remove,np.nan,inplace=True)

#閾値を決めて頻度が少ない値をNaNに置き換える
df_c6 = pd.DataFrame(train['C6'])
threshold = 120000
value_counts = df_c6.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_c6.replace(to_remove,np.nan,inplace=True)


#閾値を決めて頻度が少ない値をNaNに置き換える
df_I11 = pd.DataFrame(train['I11'])
threshold = 40000
value_counts =df_I11.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_I11.replace(to_remove,np.nan,inplace=True)


#閾値を決めて頻度が少ない値をNaNに置き換える
df_I12 = pd.DataFrame(train['I12'])
threshold = 60000
value_counts =df_I12.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_I12.replace(to_remove,np.nan,inplace=True)

#閾値を決めて頻度が少ない値をNaNに置き換える
df_I13 = pd.DataFrame(train['I13'])
threshold = 500
value_counts =df_I13.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_I13.replace(to_remove,np.nan,inplace=True)



#閾値を決めて頻度が少ない値をNaNに置き換える
df_I14 = pd.DataFrame(train['I14'])
threshold = 500
value_counts =df_I14.stack().value_counts()
to_remove = value_counts[value_counts <=threshold].index
df_I14.replace(to_remove,np.nan,inplace=True)


# In[ ]:


#データを統合する
df_new = pd.merge([df_c1,df_c2,df_c3,df_c4,df_c5,df_c6,train['I6'],train['I7'],train['I8'],train['I9'],
train['I10'],train['I11'],train['I12'],train['I13'],train['I14']])

print(df_new.head())

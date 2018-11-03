
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[186]:


country = ['Hong Kong','Singapore', 'Indonesia', 'India', 'Malaysia', 'Philippines', 'Thailand', 'Taiwan','Japan']


# In[10]:


def World_Bank_Modification(df):
    indicator = df['Indicator Name'].unique()[0]
    df = df.drop(columns=['Indicator Name', 'Country Code', 'Indicator Code'])
    df = df.set_index('Country Name').stack().reset_index().rename(columns={'Country Name':'country', 'level_1':'year',0:indicator})
    df.year = pd.to_numeric(df.year, errors='coerce')
    df.year = pd.to_datetime(df['year'],format='%Y')
    #国名修正
    df.country = df.country.replace('Hong Kong SAR, China', 'Hong Kong')
    df.country = df.country.replace('Korea, Rep.', 'South Korea')
    return df[df.country.isin(country)].set_index(['country','year'])


# In[11]:


path =['manufacturing_value_added.csv', 'import.csv','export.csv','gdp_deflator.csv']


# In[12]:


#######ここから
##Gross だけ初めから入れておく
iterables = [country, pd.to_datetime(np.arange(1960,1997),format='%Y')]
index = pd.MultiIndex.from_product(iterables, names=['country', 'year'])
df_w = pd.DataFrame({'value':np.nan},index=index)


##他を読み込んで統合
for file in path:
    a = pd.read_csv(file,skiprows=4)
    df_w = pd.merge(df_w, World_Bank_Modification(a),left_index=True, right_index=True, how='left')
    
##最初に入れたNANをDrop
df_w = df_w.drop(columns='value')
# df_w.head(20)


# In[14]:


#column名変更
df_w.columns = ['LP', 'M', 'X','DEFLATOR'] 


# In[187]:


df = df_w.reset_index()


# In[188]:


df = df[df.country=='Japan'].set_index('country')


# In[189]:


##データ整形
#1987年で調整
# df.loc[:]['LP'] = df.loc[:]['LP'] / df.loc[df.year=='1987-01-01']['LP']
# df.loc[:]['M'] = df.loc[:]['M'] / df.loc[df.year=='1987-01-01']['M']
# df.loc[:]['X'] = df.loc[:]['X'] / df.loc[df.year=='1987-01-01']['X']
df.loc[:]['DEFLATOR'] = df.loc[:]['DEFLATOR'] / df.loc[df.year=='1987-01-01']['DEFLATOR']
#Log Form
#Labor Productivityはインフレ率を考慮
df.loc[:]['LP'] = np.log(df.loc[:]['LP']/df.loc[:]['DEFLATOR'])
df.loc[:]['M'] = np.log(df.loc[:]['M'])
df.loc[:]['X'] = np.log(df.loc[:]['X'])


# In[190]:


df


# In[177]:


df = df.drop(['DEFLATOR'],axis=1)


# In[178]:


df = df.reset_index().set_index(['country','year'])


# In[179]:


df = pd.merge(df, df.pct_change(),left_index=True, right_index=True)


# In[180]:


df.columns = ['LP','M','X','dif_LP','dif_M','dif_X']


# In[181]:


df = df.reset_index().set_index(['country']).dropna()


# In[182]:


df


# In[153]:


# multiple line plot
plt.plot(df.year, df.LP, color='skyblue', label='Labor Productivity')
plt.plot(df.year, df.M, color='olive', label='Import')
plt.plot(df.year, df.X, color='olive', label='Export')
plt.legend()
plt.show()


# In[163]:


# multiple line plot
plt.plot(df.year, df.dif_LP, color='skyblue', label='Dif Labor Productivity')
plt.plot(df.year, df.dif_M, color='olive', label='Dif Import')
plt.plot(df.year, df.dif_X, color='olive', label='Dif Export')
plt.legend()
plt.show()


# In[106]:


##Unit Root Test
from statsmodels.tsa.stattools import adfuller, kpss


# In[165]:


def check_for_stationarity(X, reg='ct'):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X, regression=reg)
    print(pvalue)


# In[166]:


check_for_stationarity(df.LP,'c')
check_for_stationarity(df.M,'ct')
check_for_stationarity(df.X,'ct')
check_for_stationarity(df.dif_LP,'nc')
check_for_stationarity(df.dif_M,'nc')
check_for_stationarity(df.dif_X,'nc')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import gc
import warnings
from pandas.plotting import register_matplotlib_converters
import datetime
from pandas.tseries.offsets import MonthEnd
import pydlm.plot.dlmPlot as dlmPlot
from pydlm import dlm, trend, seasonality, longSeason, autoReg
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed 
from pydlm import dynamic
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
from sklearn.metrics import mean_squared_log_error as msle
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from itertools import product
import ast
import csv
import statistics 
from scipy.signal import savgol_filter


# In[2]:


def holiday_feat(df,windows=7, order=2):
    df['flag_'] = savgol_filter(df['flag'], windows, order, mode='interp')
    df['flag_'] = np.where( df['flag_'] < 0.0, 0.0, df['flag_'])
    return  df['flag_'].interpolate()


# In[3]:


train = pd.read_csv('train.csv')
holiday = pd.read_csv('holiday.csv')


# In[4]:


train['periode'] = pd.to_datetime(train['periode'], format="%Y/%m/%d")
time = pd.date_range(train['periode'].max() + datetime.timedelta(days=1), periods=31, freq='D')
test = pd.DataFrame(time, columns=['periode'])


# In[5]:


holiday_col = 'red date'
holiday[holiday_col] = pd.to_datetime(holiday[holiday_col], format="%d/%m/%Y")
holiday['flag'] = 1
holiday = holiday[['flag', holiday_col]].dropna()
train.set_index('periode',inplace=True)
test.set_index('periode',inplace=True)


# In[6]:


start = train.index[0]
end = train.index[-1]
ll = pd.date_range(start, end).tolist()
missing = [(i,j) for i,j in enumerate(ll) if j not in train.index]


# In[7]:


for d in missing:
    tmp = (train.iloc[d[0]] + train.iloc[d[0]-1]) / 2
    tmp.name = d[1]
    train = train.append(tmp)
train.sort_index(inplace=True)


# In[8]:


test['dataset'] = 'testing'
train['dataset'] = 'training'


# In[9]:


holiday.set_index(holiday_col,inplace=True)
train = train.join(holiday)
train['flag']=train['flag'].fillna(0)
test = test.join(holiday)
test['flag']=test['flag'].fillna(0)


# In[10]:


train = pd.concat([train,test]).fillna(0)


# In[11]:


train['flag_'] = holiday_feat(train,windows=7, order=1)


# In[12]:


train.reset_index(inplace=True)


# In[13]:


dt_col = "periode"
day_col = 'd'


# In[14]:


train['d'] = train.shape[0] - (train['periode'].max() - train['periode'] )/np.timedelta64(1, 'D')


# In[15]:


features_drop = ['d']

is_train = train["dataset"] == 'training' 
is_test = ~is_train
features = [i for i in train.columns.tolist() if i not in features_drop]

X_train = train[is_train][[day_col] + features]
X_test = train[~is_train][[day_col] +features]

id_date = train[~is_train]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# In[16]:


import pickle
with open('params_echannel/fbn.pkl', 'rb') as pkl:
        params_fbn = pickle.load(pkl)
with open('params_echannel/dlm.pkl', 'rb') as pkl:
        params_dlm = pickle.load(pkl)
with open('params_echannel/matrix.pkl', 'rb') as pkl:
        matrix_echn = pickle.load(pkl)


# In[17]:


def build_dlm(best_hyperparams,time_series, features):
    linear_trend = trend(degree=best_hyperparams['trend_degree'], discount=best_hyperparams['trend_discount'], 
                     name='linear_trend', w=best_hyperparams['trend_w'])
    seasonal = seasonality(period=7, discount=best_hyperparams['seasonal_discount'], name='seasonal', w=best_hyperparams['seasonal_w'])
    longSeasonal =  longSeason(data=time_series,period=best_hyperparams['lseasonal_period'],
                           stay=best_hyperparams['lseasonal_stay'],discount=best_hyperparams['lseasonal_discount'],
                           name='longSeasonal',w=best_hyperparams['lseasonal_w'])
    ar = autoReg(degree=best_hyperparams['ar_degree'], discount=best_hyperparams['ar_discount'], name='ar',w=best_hyperparams['ar_w'])
    regressor = dynamic(features=features, discount=best_hyperparams['reg_discount'], name='regressor', w=best_hyperparams['reg_w'])
    simple_dlm = dlm(time_series) + linear_trend + ar + seasonal +  longSeasonal + regressor
    return simple_dlm

def build_fbn(best_hyperparams, X_train, holiday):
    set_random_seed(0)
    public_holidays = pd.DataFrame({'event': 'holiday','ds': pd.to_datetime(holiday.reset_index()[holiday_col].tolist())})
    training = X_train.rename(columns={'periode': 'ds', 'kas_echannel': 'y'})[['ds','y','giro','deposito','kewajiban_lain','tabungan','rata_dpk_mingguan']]
    model = NeuralProphet(growth='discontinuous',
                          changepoints_range=best_hyperparams['changepoints_range'],
                          trend_reg=best_hyperparams['trend_reg'],
                          weekly_seasonality=True,
                          seasonality_mode=best_hyperparams['seasonality_mode'],
                          seasonality_reg=0.1,
                          n_lags=best_hyperparams['n_lags'],
                          ar_sparsity=1,
                          batch_size=None,
                          n_forecasts=DAYS_PRED,
                          loss_func='mse', 
                          epochs=None)

    model.add_lagged_regressor('giro',only_last_value=True)
    model.add_lagged_regressor('deposito',only_last_value=True)
    model.add_lagged_regressor('kewajiban_lain',only_last_value=True)
    model.add_lagged_regressor('tabungan',only_last_value=True)
    model.add_lagged_regressor('rata_dpk_mingguan',only_last_value=True)
      
    model = model.add_events(["holiday"],mode=best_hyperparams['events_mode'],
                             lower_window=int(best_hyperparams['lower_window']), 
                             upper_window=int(best_hyperparams['upper_window']))
    
    return model, training, public_holidays


# In[18]:


from neuralprophet import set_random_seed 
predictions = []
columns_name =[]
yhat=[]
DAYS_PRED=31
for i in range(DAYS_PRED):
    string = 'yhat'+str(i+1)
    yhat.append(string)
    
for idx,m in enumerate ([params_dlm,params_fbn]):
    if idx==0:
        integer = ['ar_degree','ar_w','lseasonal_period','lseasonal_stay','lseasonal_w','reg_w','seasonal_w','trend_degree','trend_w']
        for i in integer:
            m[i]=int(m[i])
        train['flag_'] = holiday_feat(train,windows=m['window'], order=m['order'])
        train['target'] = np.log1p(train['kas_echannel'])
        train1 = train.iloc[0:].reset_index()
        X_train=train1.iloc[:-31]
        X_test=train1.iloc[-31:]
        variables = train1[['flag_']].columns.tolist()
        time_series = train1['target'].values.tolist()
        features = [[train1[variables[j]].iloc[i] for j in range(0, len(variables)) ] for i in range(len(time_series))]
        features_ = [[X_test[variables[j]].iloc[i] for j in range(0, 1) ] for i in range(31)]
        featureDict_31day = {'flag_':features_}
        model = build_dlm(m,time_series, features)
        model.fit()
        predictions.append(pd.DataFrame(np.expm1(model.predictN(N=31, date=427, featureDict=featureDict_31day)[0]),columns=[str(idx)]))
    else :
        integer = ['lower_window','upper_window','n_lags']
        for i in integer:
            m[i]=int(m[i])    
        X_train = train[train['dataset']=='training']
        model, training, public_holidays = build_fbn(m, X_train, holiday)
        trn = model.create_df_with_events(training, public_holidays)
        metrics = model.fit(trn, freq="D", validate_each_epoch=False, valid_p=0.0, plot_live_loss=False)
        future = model.make_future_dataframe(trn,public_holidays,periods=DAYS_PRED,n_historic_predictions=False)
        forecast = model.predict(future)
        pred = pd.DataFrame(np.diag(forecast[yhat].dropna(axis=0,how='all')), index=[forecast[yhat].dropna(axis=0,how='all').columns])
        pred_inv = pd.DataFrame(pred.values.ravel().tolist(),columns=[str(idx)])
        predictions.append(pred_inv)


# In[19]:


pred_test = pd.concat(predictions,axis=1)


# In[20]:


echannel_preds = pd.DataFrame(np.sum(np.multiply(matrix_echn, pred_test.values),axis=1)/matrix_echn.shape[1])


# In[21]:


models_kantor = os.listdir('params_kantor/model/')


# In[22]:


params=[]
for file in models_kantor :
    path = os.path.join('params_kantor/model/',file)
    with open(path, 'rb') as pkl:
        params =  pickle.load(pkl)


# In[23]:


predictions = []
columns_name =[]
for g,idx in enumerate(params) : 
    print(g)
    best_hyperparams  = idx
    integer = ['ar_degree','ar_w','lseasonal_period','lseasonal_stay','lseasonal_w','reg_w','seasonal_w','trend_degree','trend_w']
    for i in integer:
        best_hyperparams[i]=int(best_hyperparams[i])
    train['flag_'] = holiday_feat(train,windows=best_hyperparams['window'], order=best_hyperparams['order'])
    train1 = train.iloc[0:].reset_index()
    X_train=train1.iloc[:-31]
    X_test=train1.iloc[-31:]
    time_series = X_train['kas_kantor'].values.tolist()
    variables = train1[['flag_']].columns.tolist()
    time_series = train1['kas_kantor'].values.tolist()
    features = [[train1[variables[j]].iloc[i] for j in range(0, len(variables)) ] for i in range(len(time_series))]
    features_ = [[X_test[variables[j]].iloc[i] for j in range(0, 1) ] for i in range(31)]
    featureDict_31day = {'flag_':features_}
    model = build_dlm(best_hyperparams, time_series,features)
    model.fit()
    predictions.append(pd.DataFrame(model.predictN(N=31, date=427, featureDict=featureDict_31day)[0],columns=[str(g)]))


# In[24]:


with open('params_kantor/matrix.pkl', 'rb') as pkl:
        matrix_kantor = pickle.load(pkl)


# In[25]:


pred_test = pd.concat(predictions,axis=1)
kantor_preds = pd.DataFrame(np.sum(np.multiply(matrix_kantor, pred_test.values),axis=1)/matrix_kantor.shape[1])


# In[26]:


a = pd.concat([kantor_preds, echannel_preds]).reset_index(drop=True).rename(columns={0:'value'})
a.index.rename('index', inplace=True)


# In[27]:


a.to_csv('final_submission_2.csv')


# In[ ]:





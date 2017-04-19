# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:30:06 2017

@author: Victor Suárez Gutiérrez
Research Assistant. Data Scientist at URJC/HGUGM.
Contact: ssuarezvictor@gmail.com
"""


###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Inicialization:
#%reset -f


# Define working directory
import os
os.chdir('C:/Users/Victor/Documents/Ambito_profesional/proyectos/wbdevelopment/world-development-indicators')


# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold as kfold
from sklearn import preprocessing as pps
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LassoCV as lassocv
import matplotlib.pyplot as plt
    


###############################################################################
###############################################################################
# Load Dataset:  
Country = pd.read_csv('Country.csv')
CountryNotes = pd.read_csv('CountryNotes.csv')
Footnotes = pd.read_csv('Footnotes.csv')
Indicators = pd.read_csv('Indicators.csv')
Series = pd.read_csv('Series.csv')
SeriesNotes = pd.read_csv('SeriesNotes.csv')
inputdataset = Indicators.copy()
inputdataset = inputdataset.drop(['CountryName','IndicatorName'], axis=1)  



###############################################################################
###############################################################################
# Data preprocessing:
nans = inputdataset['Value'].isnull().values.any()      # Existence of NANs.
n_countries = inputdataset['CountryCode'].unique().shape[0] 
n_years = inputdataset['Year'].unique().shape[0]
n_features = inputdataset['IndicatorCode'].unique().shape[0]
Spain_code = np.array(Country.ix[Country['TableName']=='Spain','CountryCode'])[0]
output = inputdataset.ix[(inputdataset['CountryCode']==Spain_code) & (inputdataset['IndicatorCode']=='SP.DYN.LE00.IN'), ['Year','Value']].reset_index().drop('index',1)
nans = output['Value'].isnull().values.any()      # Existence of NANs.      
features = pd.pivot_table(inputdataset.ix[(inputdataset['CountryCode']==Spain_code) & (inputdataset['IndicatorCode']!='SP.DYN.LE00.IN'), ['Year','IndicatorCode','Value']],
            values='Value', index='Year', columns='IndicatorCode')     # rows: Year; columns: Code=CountryCode+IndicatorCode.
                     
# Nans analysis:
nans = features.isnull().sum(axis=0)    # Existence of NANs for each column.
features = features[nans[nans<round(2.0*features.shape[0]/3)].index]
ending = output.ix[len(output)-1,'Year']
features = features[features.index<=ending]
nans = features.isnull().sum(axis=1)    # Existence of NANs for each row.
features = features.ix[nans[nans<round(2.0*features.shape[1]/3)].index,:]
nans = features.isnull().sum(axis=0)    # Existence of NANs for each row.
features = features[nans[nans==0].index]
starting = features.index[0]
features = features.reset_index().drop(['Year','SP.DYN.LE00.FE.IN','SP.DYN.LE00.MA.IN','SP.DYN.TO65.FE.ZS','SP.DYN.TO65.MA.ZS'],1)
output = output.ix[output['Year']>=starting,:].reset_index().drop(['index','Year'],1)
del inputdataset,nans,Country,CountryNotes,Footnotes,Series,SeriesNotes,Spain_code,n_countries,n_features,n_years



###############################################################################
###############################################################################
# Normalization and Features Selection:
modas = pd.DataFrame(np.zeros((features.shape[1],100)))
r2_cv = pd.Series(np.zeros((100)))
r2_test = pd.Series(np.zeros((100)))
for j in range(100):   
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=31+j)
    X_train = X_train.reset_index().drop('index',1)
    y_train = y_train.reset_index().drop('index',1)
    normalizer = pps.StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler = normalizer.fit(X_train)
    X_train_norm = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
    model = lassocv(n_alphas=100, alphas=None, cv=5, max_iter=10000, n_jobs=-1).fit(X_train_norm,y_train['Value'])  
    modas.ix[np.nonzero(model.coef_)[0],j] = 1        
    del model    

features = features.ix[:,modas[modas.sum(axis=1)>80].index]

# Features selection with lasso.
X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=31+j)
X_train = X_train.reset_index().drop('index',1)
y_train = y_train.reset_index().drop('index',1)
normalizer = pps.StandardScaler(copy=True, with_mean=True, with_std=True)
scaler = normalizer.fit(X_train)
X_train_norm = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
model = lassocv(n_alphas=100, alphas=None, cv=5, max_iter=10000, n_jobs=-1).fit(X_train_norm,y_train['Value'])  
coef = model.coef_
coef_index = np.argsort(coef)[::-1]
 
# As linear model is designed, drop colinearities is a must. Threshold = 0.1. Apply mutual information is not necessary.
mcov = X_train_norm.cov()
X_train_norm = pd.concat([X_train_norm.ix[:,coef_index[0]], X_train_norm.drop(X_train_norm.ix[:,np.abs(mcov.ix[:,coef_index[0]])>0.1], 1)], 1)     
features = features.ix[:,X_train_norm.columns]
name = Indicators[Indicators['IndicatorCode']==features.columns[0]]['IndicatorName'].unique()[0]
for i in range(features.shape[1]):
    print (Indicators[Indicators['IndicatorCode']==features.columns[i]]['IndicatorName'].unique()[0])



###############################################################################
###############################################################################
# Linear model:
r2_cv = pd.Series(np.zeros((100)))
r2_test = pd.Series(np.zeros((100)))
for j in range(100):
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=31+j)
    X_train = X_train.reset_index().drop('index',1)
    y_train = y_train.reset_index().drop('index',1)
    
    normalizer = pps.StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler = normalizer.fit(X_train)
    X_train_norm = pd.DataFrame(scaler.transform(X_train))
    X_test_norm = pd.DataFrame(scaler.transform(X_test))

    kf = kfold(n_splits=4, random_state=31+j)
    r2 = pd.Series(np.zeros((kf.n_splits)))
    i=0
    for train_index, validation_index in kf.split(X_train_norm):
        X_tra, X_val = X_train_norm.ix[train_index,:], X_train_norm.ix[validation_index,:]
        y_tra, y_val = y_train.ix[train_index], y_train.ix[validation_index]  
        model = lr(n_jobs=-1).fit(X_tra, y_tra['Value'])
        r2[i] = model.score(X_val, y_val['Value'])    # R^2.
        i +=1
    r2_cv[j] = r2.mean()
    r2_test[j] =  model.score(X_test_norm, y_test['Value'])   

prediction = model.predict(X_test_norm)    
                          
                          
                          
###############################################################################
###############################################################################
# Representation:
plt.figure()
plt.scatter(range(len(prediction)),prediction,c='r')
plt.scatter(range(len(prediction)),y_test['Value'],c='g')
plt.title('Total accuracy')
plt.legend(['Estimation','Real'])
print ('Total Accuracy: %0.4f%%' % (r2_test.mean()*100))            
    
plt.figure()
plt.plot(range(1,101), r2_test)
plt.plot(range(1,101), np.tile(r2_test.mean(),len(r2_test)), c='r')
plt.plot(range(1,101), np.tile(r2_test.mean()+r2_test.std(),len(r2_test)), c='r', ls='--')
plt.plot(range(1,101), np.tile(r2_test.mean()-r2_test.std(),len(r2_test)), c='r', ls='--')
plt.ylim(0.84,1)
plt.xlim(1,100)
plt.title('Test accuracy')
plt.legend(['Acc','Mean','Std'])

plt.figure()
plt.plot(range(1,101), r2_cv)
plt.plot(range(1,101), np.tile(r2_cv.mean(),len(r2_cv)), c='r')
plt.plot(range(1,101), np.tile(r2_cv.mean()+r2_cv.std(),len(r2_cv)), c='r', ls='--')
plt.plot(range(1,101), np.tile(r2_cv.mean()-r2_cv.std(),len(r2_cv)), c='r', ls='--')
plt.title('Cross validation accuracy')
plt.legend(['Acc','Mean','Std'])
plt.ylim(0.84,1)
plt.xlim(1,100)



###############################################################################
###############################################################################
# Set final model:
normalizer = pps.StandardScaler(copy=True, with_mean=True, with_std=True)
scaler = normalizer.fit(features)
X_norm = pd.DataFrame(scaler.transform(features),columns=features.columns)
model = lr(n_jobs=-1).fit(X_norm, output['Value'])



######## 
# Clear:
del Indicators,X_test,X_test_norm,X_tra,X_train,X_train_norm,X_val,mcov, \
i,j,train_index,validation_index,y_test,y_tra, \
y_train,y_val,coef_index,modas,r2



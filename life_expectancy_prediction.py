"""
Created on Wed 30 23:24:47 2017

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
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf as acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf as pacf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
    


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
output = inputdataset.ix[inputdataset['IndicatorCode']=='SP.DYN.LE00.IN', ['CountryCode','Year','Value']]
input_features = inputdataset.copy()
input_features = input_features.ix[input_features['IndicatorCode']!='SP.DYN.LE00.IN',['CountryCode','Year','IndicatorCode','Value']]


    
###############################################################################
###############################################################################
# Time Series Analysis: 
Spain_code = np.array(Country.ix[Country['TableName']=='Spain','CountryCode'])[0]
life_expectancy = output.ix[output['CountryCode']==Spain_code,['Year','Value']]
life_expectancy = life_expectancy.reset_index().drop('index',1)
serie = life_expectancy['Value']

# Differences:
serie_diff = serie - serie.shift()
serie_diff = serie_diff.dropna().reset_index().drop('index',1)

result = adfuller(serie_diff['Value'], autolag='AIC')
print('p-value: %f' % result[1])    # No leave H0.

X_aux = (serie_diff-serie_diff.shift()).dropna().reset_index().drop('index',1)
X = X_aux-X_aux.mean()

result = adfuller(X['Value'], autolag='AIC')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])    # Leave H0.
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

if result[1]<0.01:
    print ('La serie temporal es estacionaria con un p-valor de: %.9f' % result[1])


# Testing phase:
X_train = X.ix[:len(X)-5,'Value']
X_test = X.ix[len(X_train):,'Value']

ACF = acf(X_train, nlags=40)
plot_acf(ACF, alpha=0.05)
PACF = pacf(X_train, nlags=40)
plot_pacf(PACF, alpha=0.05)    


p=0
q=0
model = ARIMA(np.array(X_train), order=(p,2,q)).fit()
aic = model.aic
for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(np.array(X_train), order=(p,2,q)).fit()
        except:
            print('ARIMA Model p=%i and q=%i is not consistent.' % (p,q))
        if model.aic<aic:
            aic=model.aic
            p_ = p
            q_ = q

# Prediction with best model (lower AIC):
model = ARIMA(np.array(X_train), order=(p_,2,q_)).fit()
prediction = model.predict(start=len(X_train), end=len(X)-1)  
plt.figure()
plt.plot(life_expectancy.ix[2:,'Year'], X)
plt.plot(life_expectancy.ix[2:,'Year'], np.concatenate((np.array(X_train),prediction)))
plt.title('Life expectancy non-stationary')
plt.legend(['Real','Prediction'])
Error_non_stationary = np.abs(np.concatenate((np.array(X_train),prediction))-np.array(X['Value']))[-4:]
Erroraverage_non_stationary = Error_non_stationary.mean()
percentage_non_stationary = (Erroraverage_non_stationary/(X.max()-X.min()))*100
             
# Base time series:
final_prediction = np.concatenate((np.array(X_train),prediction))+np.tile(X_aux.mean(),len(np.concatenate((np.array(X_train),prediction))))
final_prediction2 = final_prediction+np.array(serie_diff.shift().ix[1:,'Value'])
final_prediction3 = final_prediction2+np.array(serie.shift())[2:]
plt.figure()
plt.plot(life_expectancy.ix[2:,'Year'], serie[2:],c='b')
plt.plot(life_expectancy.ix[2:,'Year'], final_prediction3,c='r')
plt.title('Life expectancy')
plt.legend(['Real','Prediction'])


# Error metrics:
Error = np.abs(final_prediction3-np.array(serie[2:]))[-4:]
Erroraverage = Error.mean()
percentage = (Erroraverage/(serie[2:].max()-serie[2:].min()))*100                   
print ('Error mean percentage at non-stationary time series is %0.4f%%' % percentage_non_stationary)
print ('Error mean percentage at original time series is %0.4f%%' % percentage)



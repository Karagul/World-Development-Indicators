"""
Created on Sat 1 13:25:47 2017

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


    
###############################################################################
###############################################################################
# Time Series Analysis: 
Spain_code = np.array(Country.ix[Country['TableName']=='Spain','CountryCode'])[0]
life_expectancy = output.ix[output['CountryCode']==Spain_code,['Year','Value']]
life_expectancy = life_expectancy.reset_index().drop('index',1)
serie = life_expectancy['Value']
X_train = serie[:len(serie)-4]
X_test = serie[len(X_train):]

plt.figure()
plt.plot(X_train)     # linear model.

# lineal model estimation.
A = np.array([life_expectancy.ix[range(len(X_train)),'Year'], np.ones(len(X_train))])
y = np.array(X_train)
w = np.linalg.lstsq(A.T,y)[0] # obtaining the parameters
line = w[0]*life_expectancy.ix[range(len(X_train)),'Year']+w[1] # regression line
plt.figure()
plt.plot(life_expectancy.ix[range(len(X_train)),'Year'],line,'r-',life_expectancy.ix[range(len(X_train)),'Year'],y,'o')
plt.title('Life expectancy')

# Prediction with Linear model:
prediction = w[0]*life_expectancy['Year']+w[1]
plt.figure()
plt.plot(life_expectancy['Year'],life_expectancy['Value'],'b',life_expectancy['Year'],prediction,'r-')
plt.title('Spain life expectancy')

# Error metrics:
Error = np.abs(prediction[-4:]-np.array(life_expectancy.ix[len(life_expectancy)-4:,'Value']))
Erroraverage = Error.mean()
percentage = (Erroraverage/(life_expectancy['Value'].max()-life_expectancy['Value'].min()))*100                   
print ('Error mean percentage is %0.4f%%' % percentage)

# Representation:
plt.figure()
plt.plot(life_expectancy['Year'],life_expectancy['Value'],'b')
plt.plot(life_expectancy.ix[:len(life_expectancy)-4,'Year'],prediction[:-3],'r--')
plt.plot(life_expectancy.ix[len(life_expectancy)-4:,'Year'],prediction[-4:],'g--')
plt.title('Spain')
plt.xlabel('Year')
plt.ylabel('Life expectancy')


# Future checking:
output2015 = w[0]*2015+w[1]
real2015 = 82.8
future_error = 100*(np.abs(real2015-output2015)/real2015)



"""
Created on Mon Feb 13 21:13:24 2023

@author: daviddeleon
"""

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import missingno as msno


"import data"
housingDataSet = pd.read_csv("housing.csv")

df = housingDataSet

df.head

descriptionOfData = df.describe()

infoOfData = df.info()


#check for missing values


missingValues = df.isnull().sum()

missingValues

df.count()

cleanedDf = df.dropna()
#List out the different categorical variables in the 
ocean_proximity = cleanedDf['ocean_proximity'].unique()
#one-hot encoding

cleanedDf['ocean_proximity'] = cleanedDf['ocean_proximity'].map({1:'NEAR BAY', 2: '<1H OCEAN', 3: 'INLAND',4: 'NEAR OCEAN',5: 'ISLAND'})

cleanedDf = pd.get_dummies(cleanedDf, columns=['ocean_proximity'], prefix = '', prefix_sep = '')
cleanedDf


# dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
# dataset.tail()



import plotly.express as px

df2=df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income','median_house_value']]
fig = px.scatter_matrix(df2,dimensions=['housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income','median_house_value'])

fig.update_traces(marker=dict(size=1, line=dict(width=1, color="DarkSlateGrey")))
fig.update_traces(diagonal_visible=False)
fig.show()

'''
# Random Forest Regression with Enseble Bagging
# Importing the libraries
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''# Importing the dataset'''
df = pd.read_csv('allocations.csv')
X1 = df.iloc[:, 0:3]
X2=  df.iloc[:, 3:4]
y = df.iloc[:, -1]

'''
# Encoding the Independent Variable Analyst Level, Complexity and data
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X1=X1.apply(LabelEncoder().fit_transform)

'''
#Adding the encoded data and the Height column and concatenating the X1 and X2 arrays
'''
X=pd.concat((X1,X2),axis = 1)

#Deleting the intermediate frames, these would no longer be used in the subsequent code
del X1,X2

'''
lets try and fit on 100 trees, the value would be more close to the actual
96.32% accuracy is a better picture as compared to 10 trees
'''
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0,oob_score=True)
regressor.fit(X, y)


print(regressor.oob_score_)



'''
#1.Data processing, below are the libraries we that we would need for all ML sessions
'''
import pandas as pd

'''
# Importing the dataset, remember to set a working directory in Spyder
#Separating data in X(country, age and sal) and Y(Purchased)
#For printing you can use the console as well
'''

df = pd.read_csv('Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values


'''
# Taking care of missing data, Imputer is a class from the sklearn library
# The missing data in above df would be stored as  'nan'
'''
from sklearn.preprocessing import Imputer

'''
#imputer, we are creating an object from the Imputer class
#median is one of the strategy, the default strategy is mean
#axis =0 is the mean of columns and 1 is the mean of rows
#Ctrl +I for inspect and you can check the documentation in Python

'''
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
type(imputer)
print(imputer)

'''
#Fitting Imputer to our X, we are doing this for all rows and just columns 1 and 2, upperbound 3 is excluded :)

'''
impuer=imputer.fit(X[:,1:3])
'''
#Applying method transform for pushing the median values

'''
X[:, 1:3] = imputer.transform(X[:, 1:3] )




# What would be the price for a home that is 3300 square feet in area, this is an example of the interpolation
#Importing various modules and reading csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('housing.csv')

#iloc for reading salary(dependent) and yearexperience(independent) into arrays
# Reading data into X and Y variables, pay close attention on the dimension
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values


#SPlitting the data and lets check for the predicted value
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0,random_state=0)


#Implement classifier based on Simple Linear Regression, creating the object, fitting and then predicting
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)
#Estimating the price for a 3300 square feet home, the value come out to be 637197, 
#This is between 3200 sq feet and 3600 sq feet available price :)
# Why the predicted value came out to be same for split and without split
#I think now you are able to understand when and where to split the data
Y_predict_val=reg.predict(3300)
print(Y_predict_val)
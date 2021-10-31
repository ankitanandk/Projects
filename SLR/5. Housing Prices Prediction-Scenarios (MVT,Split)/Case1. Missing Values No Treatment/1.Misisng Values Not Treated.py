#The csv now contains missing values of the "Y" variable at 2 instances :(
#Let's see what would be the outcome if we do not treat the missing values
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('housing.csv')

#iloc for reading salary(dependent) and yearexperience(independent) into arrays
# Reading data into X and Y variables, pay close attention on the dimension
X=df.iloc[:,:-1].values
# Read it like a numpy array
Y=df.iloc[:,1:2].values

#Implement classifier based on Simple Linear Regression, creating the object, fitting and then predicting
#The missing values are not well expected by the model, thus we definitely need to handle them in our code !!
# I hope you get, why we need the MVT as the value error is raised
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)

############################################################################################################


                        '''FINAL MODEL BY KEEPING R&D SPEND AND MARKETING SPEND'''

############################################################################################################
# Data Preprocessing Template for help in a standard format
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset from the working library
# Reading the first 2 variables to X 
#( X is an object as it is a mix of diff types of variables)and last to y, thats why we cannot view it, 
#you can print in console and have a look at the matrix
df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, [0,2]].values
y = df.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set, in the same 80:20 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Fitting Multiple Linear Regression to the Training set of X and y :)
# This is still a linear regression model, just with many Xs
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#Predicting the test results, now just compare the predicted results with y_test
# The y_test is the actual values and the y_pred are predicted ones, y_test is used to comapre with y_pred for residual Analysis
# Honestly at this stage,we have a model that can predict profit from amount spent on R&D ,Admin & Marketing
# The Machine has learned and now we can make a prediction for the new set of data
y_pred=reg.predict(X_test)

#The R square seems to be very good for this model, the model explains 94.96% 
# The adjusted R square has increased and thus this is the best fit model after the variable selection, congrats :)
r2_score=reg.score(X_train,y_train)
print(r2_score)

############################################################################################################


                        '''NEW DATA with 10 new comapnies for multivariate prediction'''

############################################################################################################

# Lets say now,we have been given data for 10 companies on R&D ,Administration and Marketing and predict profit
# I have picked top 10 rows and the values are substracted by 1k,2k and 3k respectively 
#last row same with profit 149760 for testing purposes
# We are assuming that the last model reg is run prior to this and it is ready
# Importing the dataset from the working library
# Reading the first 2 significant variables to X,no need of Y, as we would predict profit from the already built model
df = pd.read_csv('10_Startups_newdata.csv').values
Xnew = df[:, [0, 2]]

#Predicting the results for the 10 new comapnies
# Splitting is not required as the model is already trained and ready
# The new profit values are ready for the 10 new companies, here we have used the multivariate concept for predicting Y
y_new=reg.predict(Xnew)




    
















#The most important scenario, we are now using reg_new model on salary_new
# We have got the new salary values of the hired employees and we have to now adjust the model as per new available data
#This is model corerction and model validation in light of new data points for adjusting the OLS best fit line
#Salary and salary_new_actuals would be used for training the model again for a better fit and smart model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('salary.csv')
df_actuals=pd.read_csv('salary_new_actuals.csv')

#Appending the 2 frames for making a new data frame.
#we would train model on this appended data, the model would now learn again and adjust itself for the new set
# Now we have 40 data points to make the model learn again, this would improve the model as per available data
df_new2=pd.concat([df,df_actuals])

# Reading data into X and Y variables
X=df_new2.iloc[:,:-1].values
Y=df_new2.iloc[:,1].values

'''
#splitting data into training and test sets
# Do we really need to split the data? think :)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
'''
#Implement the reg object based on Simple Linear Regression
#Why are we fiiting and creating the model again, think :)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)

#Reading new salary data,(4 data points)for which we have to predict based on salary.csv and salary_new_actuals.csv
df_new_latest=pd.read_csv('salary_new2.csv')
X_test_latest=df_new_latest.iloc[:].values

#predicting new salary by using the earlier trained model
Y_predict_latest=reg.predict(X_test_latest)

#look for 11 yr exp sal was 129621 previously, however as per last model now it is 129613
#the model has adjusted itself to the new available data and now the sal is lowered down a bit :)
print(X_test_latest,Y_predict_latest)

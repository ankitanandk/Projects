# What would be the price for a home that is 3300 square feet in area, this is an example of the interpolation
#Importing various modules and reading csv
# The csv now contains missing values of the "Y" variable at 2 instances :(
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sb
df=pd.read_csv('housing.csv')

'''
skewness = 0 : normally distributed.
skewness > 0 : more weight in the left tail of the distribution.
skewness < 0 : more weight in the right tail of the distribution. 

As a general rule of thumb: If skewness is less than -1 or greater than 1, 
the distribution is highly skewed. If skewness is between -1 and -0.5 or between 0.5 and 1,
the distribution is moderately skewed. 
If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

we can get the skewness of each column of theframe by using the skew pandas function 

As the data is moderately skewed with a value of -0.94, we can use the mean for the MVT

'''
df.skew(axis = 0, skipna = True)

#iloc for reading salary(dependent) and yearexperience(independent) into arrays
# Reading data into X and Y variables, pay close attention on the dimension
X=df.iloc[:,:-1].values
# Read it like a numpy array
Y=df.iloc[:,1:2].values


'''
**********************************
Assumption 1 of Simple Linear Regression
1. The 2 variables X and Y need to be continous numeric, NOT categorical.
**********************************
'''
# By Inspecting the 2 variables X and Y, both are continous and Numeric
# Assumption 1 PASS

'''
**********************************
Assumption 2 of Simple Linear Regression
2. Data is free of Missing values, if missing values are there MVT need to be performed.

**********************************
'''
def mmissing_counter_amit(Y):
    if np.count_nonzero(~np.isnan(Y))==len(Y):
        return print("No missing values :)")
    else:
        return print("Missing value count is",len(Y)-np.count_nonzero(~np.isnan(Y)), "out of" ,len(Y))
    
mmissing_counter_amit(Y)

# Assumption 2 FAIL


# Taking care of missing data, Imputer is a class from the sklearn library
# The missing data in above df would be stored as  'nan'
from sklearn.preprocessing import Imputer
#imputer, we are creating an object from the Imputer class
#We need "NAN" as the imputer works with "NAN"
#Mean is also the default stratgey in Imputer, you can even ignore, it would consider mean only
#axis =0 is the mean of rows and 1 is the mean of column
#Ctrl +I for inspect and you can check the documentation in Python
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#Fitting Imputer to our Y series
imputer=imputer.fit(Y[:])

#Applying method transform for pushing the mean values into the column
Y[:] = imputer.transform(Y[:] )


'''
**********************************
Assumption 3 of Simple Linear Regression
3. Data is free of outliers, if the  outliers are there take a business decision for removal @95% z=1.96 or @99% z=2.97

**********************************
'''
z = max(np.abs(stats.zscore(Y)))
print(z)

# Assumption 3 PASS

'''
**********************************
Assumption 4 of Simple Linear Regression
4. Linear relationship between X and Y, we are using the imputed arrays below after reshaping from 1 to 0
**********************************
'''
n=len(X) 
x=X.reshape((n,))
y=Y.reshape((n,))
np.corrcoef(x,y)

# Assumption 4 PASS as the relationship seems to be very good linear :) 


'''
**********************************
Assumption 5 of Simple Linear Regression
5. Depending on the case/business need, data should be split into TRAIN and TEST
**********************************
'''
# Assumption 5 PASS as we are splitting data, as per the problem

'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Carrying on with Simple Linear REgression
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''

#SPlitting the data and lets check for the predicted value
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0,random_state=0)


#Implement classifier based on Simple Linear Regression, creating the object, fitting and then predicting
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)
#Estimating the price for a 3300 square feet home, the value come out to be 634223,this is different from preious 637197 

Y_predict_val=reg.predict(np.array(3300).reshape(1,1))
print(Y_predict_val)
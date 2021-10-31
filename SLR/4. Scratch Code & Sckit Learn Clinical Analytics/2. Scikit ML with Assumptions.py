#importing pandas,numpy, scipy and seaborn and matplotlib libraries/modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sb

#Run in spyder for better clarity and understanding
#importing the data from saved csv file, r is used with path to read the raw path and avaoid any special meaning
df=pd.read_csv('headbrain.csv')

#Keeping just X and Y for the Simple linear Regression and overwriting preious dataframe
df=df[['Head Size(cm^3)','Brain Weight(grams)']]

#collecting x and y data from the data frame to an one dimension numpy array 
X=df['Head Size(cm^3)'].values
Y=df['Brain Weight(grams)'].values

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
mmissing_counter_amit(X)

# Assumption 2 is PASS for X and Y

'''
**********************************
Assumption 3 of Simple Linear Regression
3. Data is free of outliers, 
If the  outliers are there take a business decision for removal @95% z=1.96 or @99.97% z=2.96

**********************************
'''
z = max(np.abs(stats.zscore(Y)))
print(z)

# Assumption 3 is PASS, however we need to keep an eye as there seems to be some high values exceeding 4 sigma

'''
**********************************
Assumption 4 of Simple Linear Regression
4. Linear relationship between X and Y
**********************************
'''
sb.pairplot(df)
np.corrcoef(X,Y)
df.cov()


# Assumption 4 PASS as the relationship seems to be strongly linear, corr is .7995 and covariance is also positive


'''
**********************************
Assumption/suggestion 5 of Simple Linear Regression
5. Depending on the case/business need, data should be split into TRAIN and TEST
**********************************
'''
# Assumption 5 PASS as we are NOT splitting data, as per the problem and required solution

'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Carrying on with Simple Linear Regression
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''

# Printing the dimesnion of X and it is 1, just for clarity that X and Y are now 1D arrays in Python
print(np.ndim(X))

#we are trying to attempt the use of sklearn (scikit) learn to reproduce the same results, remeber the rsquare was .63
#Below Classes are imported from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

'''Cannot use Rank 1 matrix in scikit learn Total number of values captured in  a array, basically the number of rows
Now, when you pass a one dimensional array of [n,], Scikit-learn is not able to decide that what you 
have passed is one row of data with multiple columns, or multiple samples of data with single column.
# i.e. sklearn may not infer whether its n_samples=n and n_features=1'''
n=len(X) 
X=X.reshape((n,1))
# Printing the dimesnion of X and it is 2 now, I hope this is now clear, walah :)
print(np.ndim(X))

#Creating Model, this basically creates a linear regression object from class LinearRegression
#Remember the object declaration learning , now the reg object owns all properties of the class :)
reg=LinearRegression()

#fitting training data in to the line using the fit method, fit is a method in the Clas LinearRegression
reg=reg.fit(X,Y)

#Y prediction using the predict method, the actual X are now pushed to model.
Y_pred=reg.predict(X)

#Calculating R2 score and printing it
r2_score=reg.score(X,Y)
print(r2_score)



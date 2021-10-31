#Importing various modules and reading Excel file from the folder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sb


df=pd.read_excel('parking price.xlsx')

#iloc for reading riders an price data
# Reading data into X and Y variables, if you notice X with iloc, you would not need to reshape the array 
X=df.iloc[:,5:6].values
Y=df.iloc[:,1].values

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

# Assumption 2 PASS

'''
**********************************
Assumption 3 of Simple Linear Regression
3. Data is free of outliers, if the  outliers are there take a business decision for removal @95% z=2 or @99% z=2.96

**********************************
'''
z = max(np.abs(stats.zscore(Y)))
print(z)

# Assumption 3 PASS

'''
**********************************
Assumption 4 of Simple Linear Regression
4. Linear relationship between X and Y
**********************************
'''
sb.pairplot(df)
x=df['Average parking rates per month'].values
y=df['Number of weekly riders'].values
np.corrcoef(x,y)

# Assumption 4 PASS as the relationship seems to be very good linear in the opposite direction, 
#look at -ve sign, we need to keep a close eye on Rsquare and lets see if Linear fit is the best fit :)

'''
**********************************
Assumption 5 of Simple Linear Regression
5. Depending on the case/business need, data should be split into TRAIN and TEST
**********************************
'''
# Assumption 5 PASS as we are NOt splitting data, as per the problem

'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Carrying on with Simple Linear REgression
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''

'''

#splitting data into training and test sets
# DO you think we need the spliting, if NOT why? :)
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

'''
#Creating reg object, so the steps are 1. Declare 2. Fit 3. Predict :)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)
#Predicting the median house prices as per the model for 5.45% of interest rates, as per the requirements
Y_predict=reg.predict(210)

#Implement the scatter line graph with X and Y data
plt.scatter(X,Y,color='red',label="Scatter data")
plt.plot(X,reg.predict(X),c='green',label="Reg line")
#Giving title to the plot for making it look better, also giving labels to X and Y axis
# Look at the graph and share what you conclude on the coefficient of correlation and slope :) :)
plt.title('House prices prediction')
plt.xlabel('Rate of Interest')
plt.ylabel('House price')
plt.legend()
plt.show()   

#Calculating the R square and printing it for the goodness of fit, 38.45%
# What the Rsquare tells you, why is it happening and what are the teakeaways? 
r2_score=reg.score(X,Y)
print(r2_score) 
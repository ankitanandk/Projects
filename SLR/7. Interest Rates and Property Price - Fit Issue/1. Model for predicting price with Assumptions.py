#Alwyas read the Business Sense document to understand the problem at hand and the data 
#Importing various modules for graphing and reading Excel file from the folder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sb

#Importing the excel and creating the dataframe
df=pd.read_excel('property.xlsx')

#iloc for reading rate and price data (The rate is the X and price is the Y)
# Reading data into X and Y variables, if you notice X with iloc, you would not need to reshape:)
# X is now a 2D array or a matrix and Y is a 1D array or a series
X=df.iloc[:,1:-1].values
Y=df.iloc[:,2].values

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
3. Data is free of outliers, if the  outliers are there take a business decision for removal 95% z=1.96 or @99.97% z=2.97

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
x=df['interest_rate_30years'].values
y=df['Median home price'].values
np.corrcoef(x,y)

# Assumption 4 is not PASSING good as the relationship seems to be good linear in the opposite direction, 
#look at -ve sign, we need to keep a close eye on Rsquare and lets see if Linear fit is the best fit :)
#The Simple linear may not be the right approach, we may need polynomial in this case for a better Rsquare

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
Y_predict=reg.predict(np.array(5.45).reshape(1,1))

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
# What the Rsquare tells you, why is it happening and what are the takeaways? 
# If the R square is not good, how can we improve the model?
r2_score=reg.score(X,Y)
print(r2_score) 

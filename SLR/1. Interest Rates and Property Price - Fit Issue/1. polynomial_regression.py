'''Polynomial Regression for the problem
Importing the various libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sb

'''#Importing the excel and creating the dataframe'''

df=pd.read_excel('property.xlsx')

'''iloc for reading rate and price data
Reading data into X and Y variables,  X is interest rates and y is the price

if you notice X with iloc, you would not need to reshape:)'''
X=df.iloc[:,1:-1].values
y=df.iloc[:,2].values

'''
**********************************
Assumption 1 of Simple Linear Regression
1. The 2 variables X and Y need to be continous numeric, NOT categorical.
The X can however be ordinal.Y should always be continous
**********************************
'''
'''
# By Inspecting the 2 variables X and Y, both are continous and Numeric
# Assumption 1 PASS.

'''
'''
**********************************
Assumption 2 of Simple Linear Regression
2. Data is free of Missing values, if missing values are there MVT(Missing Value Treatment) need to be performed.

**********************************
'''
def mmissing_counter_amit(y):
    if np.count_nonzero(~np.isnan(y))==len(y):
        print ("No missing values")
    else:
        print("Missing value count is",len(y)-np.count_nonzero(~np.isnan(y)), "out of" ,len(y))
    
mmissing_counter_amit(y)
mmissing_counter_amit(X)

'''# Assumption 2 PASS as no missing values in X and y'''

'''
**********************************
Assumption 3 of Simple Linear Regression
3. Data is free of outliers, 
If the  outliers are there take a business decision for removal @95% z=1.96 or @99.97% z=2.97

**********************************
'''
'''#np.abs is required as the z scores are negative as well, refer to z score notes
z for y is +ve and z for x is -ve... still in the range though

'''
z1 = max(np.abs(stats.zscore(y)))
z2 = max(np.abs(stats.zscore(X)))
print(z1,z2)

'''# Assumption 3 PASS with close range, keep an eye on the extreme values as they might influence results'''

'''
**********************************
Assumption 4 of Simple Linear Regression
4. Linear relationship between X and Y
**********************************
'''
x=df['interest_rate_30years'].values
y=df['Median home price'].values
df.cov()
np.corrcoef(x,y)
sb.pairplot(df)

'''

# Assumption 4 FAIL as the relationship seems to be curvilinear and we may need to use polynomial Regression
#The correlation and covariance is -ve, not strongly negative though

'''

'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Carrying on with Polynomial Linear Regression
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''
'''
lets plot a scatter between X and Y, to see the relationship
look at the graph and you would know why are we resorting to polynomial regression
The straight line with degree 1  would not do justice for the model as it would
not be able to fit maximum points
The line is great for a polynomial of degree 1, here the degree seems to be high
'''
plt.scatter(X, y, c= ("red"))

'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Approach 1: Let's try and fit a linear model, just to compare the outputs of both the models

Creating Linear model, though we know, this is not best suited for this problem
Let's see the power of prediction if we have gone with the linear model
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''
'''
# Fitting Linear Regression to the dataset
# No need of split and feature scaling, feature scaling is taken care by the library itself
# we are trying to comapre the results between Linear and Polynomial regression 
#Look at the rsquare 38.45 % and we would try and see if we can better this :)

'''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
r2_score=lin_reg.score(X,y)
print(r2_score)

'''# Visualising the Linear Regression results, the scatter for observed data points,The line for prediction and scatter'''
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'green')
plt.title('Linear Regression')
plt.xlabel('Interest Rates')
plt.ylabel('Price')
plt.show()

'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Approach 2: Creating Polynomial model
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''
'''
# Fitting Polynomial Regression to the dataset
# Polynomial is in the preprocessing library
'''   
from sklearn.preprocessing import PolynomialFeatures

'''
# Creating the object from the Class and the degree helps in creating the x2,x3 and so on with powers
# Note it would create column with ones,the first column, we have seen this in multiple regression for b0 :)
# The degree by default is 2, we would see later, how to decide the degree of the model
# Look at the r square it has jumped to 73.14%, this curve is now trying to fit the maximum values
'''

poly_reg = PolynomialFeatures(degree=2)
'''# This is required for creating additional polynomial terms'''
X_poly = poly_reg.fit_transform(X) 

'''# Creating Linear regression object'''
lin_reg_2 = LinearRegression() 

'''# Fitting X_poly and y to the model'''
lin_reg_2.fit(X_poly, y)
r2_score_poly=lin_reg_2.score(X_poly,y)  

'''

# Visualising the Polynomial Regression results 
# Note for the plot we are using the poly_reg.fit_transform(X), as the model would work on that
# We wish to make the X generic, else we would have used X_poly 
#X_poly is for degree 4 and we wish that the plot should be created for X
'''
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Interset Rates and Price prediction(Polynomial Regression)')
plt.xlabel('Interest Rates')
plt.ylabel('Price')
plt.show()

'''
Creating Polynomial model for Degree 3 or 4, iterate the whole process and look at the r square
If the r square starts to stay almost static, that should be the degree of the polynomial
'''

'''
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# Change degree to 3 and then 4, realize that the curvefitting gets better
# Creating the grid so that the scale zooms in and we can see the fit better
'''
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('(Polynomial Regression)')
plt.xlabel('Interest Rates')
plt.ylabel('Price')
plt.show()

'''
# Predicting a new result with Linear Regression￼
# This would try and predict the price of home with intereset rate as 5.45 Linear Regression degree 1
'''
Y_predict=lin_reg.predict(np.array(5.45).reshape(1,1))
print(Y_predict)
'''
# Predicting a new result with Polynomial Regression
# This would try and predict the home price with 5.45 intereset rate the polynomial curve, with degree 4
#The results are much better and accurate
'''
print(lin_reg_2.predict(poly_reg.fit_transform(np.array(5.45).reshape(1,1))))
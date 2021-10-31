'''
# Polynomial Regression is a case of Multiple linear regression
# Importing the libraries
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sb

'''
# Importing the dataset, we dont need position as position and Level are same, we would have a multicolinearity issue
# No need of split as the observations are too few and we need accurate predictions,and no need of scaling as well :)
# To make X look like a matrix with 2 dimension we use 1:2, remember this , we need matrix and not vector or series :)
#The X could be ordinal and not continous, the Y need to be contionous.. imp

Here the X is ordinal and this explains the above statement :0  

'''
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2:3].values

'''
**********************************
Assumption 1 of Simple Linear Regression/Polynomial
1. The 2 variables X and Y need to be continous numeric, NOT categorical.The X can however be ordinal.
Y should always be continous
**********************************
'''
'''
# By Inspecting the 2 variables X and Y, both are continous/ordinal for X and Numeric
# Assumption 1 PASS with a twist.
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
'''#np.abs is required as the z scores are negative as well, refer to z score notes'''
z1 = max(np.abs(stats.zscore(y)))
z2 = max(np.abs(stats.zscore(X)))
print(z1,z2)

'''# Assumption 3 PASS with a twist, as we can not remove the salary values of the highest order'''

'''
**********************************
Assumption 4 of Simple Linear Regression
4. Linear relationship between X and Y
**********************************
'''
x=df['Level'].values
y=df['Salary'].values
np.corrcoef(x,y)
df.cov()
sb.pairplot(df)



'''# The scater plot says it all as the relation is in a curve, if I use simple linear what would happen'''
plt.scatter(X, y, c= ("red"))
'''
# Assumption 4 FAIL as the relationship seems to be curvilinear and we may need to use polynomial Regression and NOt simple linear
#The correlation and covariance is +ve
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
#lets plot a scatter between X and Y, to see the relationship
#look at the graph and you would know why are we resorting to polynomial regression
#The line would not do justice for the model as it woul not be able to fit maximum points
#The line is great for a polynomial of degree 1, here the degree seems to be high

'''

plt.scatter(X, y, c= ("red"))


'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Creating Linear model, though we know, this is not best suited for this problem
Let's see the power of prediction if we have gone with the linear model

This is for comparison between Linear and Poly model
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''
'''
# Fitting Linear Regression to the dataset
# No need of split and feature scaling, feature scaling is taken care by the library itself
# we are trying to comapre the results between Linear and Polynomial regression 
#Look at the rsquare 66.9 % and we would try and see if we can better this :)

'''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
r2_score=lin_reg.score(X,y)
print(r2_score)


'''# Visualising the Linear Regression results, the scatter for observed data points,
The line for prediction and scatter'''


plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Creating Polynomial model for the comparison to Linear model
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''
'''
# Fitting Polynomial Regression to the dataset
# Polynomial is in the preprocessing library  


''' 

from sklearn.preprocessing import PolynomialFeatures

'''Creating the object from the Class and the degree helps in creating the x2,x3 and so on with powers
Note it would create column with ones,the first column, we have seen this in multiple regression for b0 :)
The degree by default is 2, we would see later, how to decide the degree of the model
Look at the r squae it has jumped to 91.62%, this curve is now trying to fit the maximum values

The R2 for degree=2 is 91.62%
The R2 for degree=3 is 98.12%  >> You can stop at 3 as the gain is not much and this leads to overfitting 
The R2 for degree=4 is 99.73%

'''
poly_reg = PolynomialFeatures(degree = 3)

'''Lets see what all theis object contains'''
poly_reg.__dict__

'''
It create a matrix with a constant tem and degree=2..
 so total 3 variables for the polynomial eq b0+b1x1+b2x2**2
 
The fit only fits the data and calculate the paraemters, 
Transform is used to create the many other variables with varying powers
 
Transform: would give the constant terms plus degree powers
 
'''
X_poly = poly_reg.fit_transform(X)

'''# Creating Linear regression object'''
lin_reg_2 = LinearRegression() 

'''# Fitting X_poly and y to the model'''
lin_reg_2.fit(X_poly, y)

'''Lets see the coeff stored by this object '''
lin_reg_2.__dict__


r2_score_poly=lin_reg_2.score(X_poly,y)  

'''
# Visualising the Polynomial Regression results 
# Note for the plot we are using the poly_reg.fit_transform(X), as the model would work on that
# We wish to make the X generic, else we would have used X_poly 
#X_poly is for degree 4 and we wish that the plot should be created for X
'''

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''
Creating Polynomial model for Degree 3 or 4, iterate the whole process and look at the r square
If the r square starts to stay almost static, that should be the degree of the polynomial

Keep repeating the above code for a better fit and stop at the inflexion or the elbow point
'''
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# Change degree to 3 and then 4, realize that the curvefitting gets better
# Creating the grid so that the scale zooms in and we can see the fit better
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


'''Result comparison of Linear Regression VS Polynomial Regression, this is part ofthe problem
where we are trying to see the salary of a 6.5 year expereinced employee and he is close to 160k
 '''

# Predicting a new result with Linear Regression￼
# This would try and predict the sal of a 6.5 level employee using the straight line, Linear Regression degree 1
print(lin_reg.predict(np.array(6.5).reshape(1,1)))

'''
# Predicting a new result with Polynomial Regression
# This would try and predict the sal of a 6.5 level employee using the polynomial curve, with degree 4
#The results are much better and accurate
'''
print(lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,1))))
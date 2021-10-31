'''Polynomial Regression
Importing the various libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sb
'''
#Importing the excel and creating the dataframe
'''
df=pd.read_excel('property.xlsx')

'''iloc for reading rate and inflation for X and price data for Y
Reading data into X and Y variables, if you notice X with iloc, you would not need to reshape:)
X: There are 2 X now Intereset rates and the Inflation

'''
X=df.iloc[:,1:3]
y=df.iloc[:,2]

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

'''# Assumption 2 PASS as no missing values in X and y'''

'''
**********************************
Assumption 3 of Simple Linear Regression
3. Data is free of outliers, 
If the  outliers are there take a business decision for removal @95% z=1.96 or @99.97% z=2.97

**********************************
'''
'''#np.abs is required as the z scores are negative as well, refer to z score notes'''
z = max(np.abs(stats.zscore(y)))
print(z)

'''# Assumption 3 PASS with close range, keep an eye on the extreme values as they might influence results'''

'''
**********************************
Assumption 4 of Simple Linear Regression
4. Linear relationship between X and Y
Interest rates are inversly proportional to Home Price.
Interest rates go down the prices go up as more money is available
 
 Inflation rates are proportional to Home Price
**********************************
'''

df.cov()
sb.pairplot(df)

'''
# Assumption 4 FAIL as the relationship seems to be curvilinear and we may need to use polynomial Regression
#The correlation and covariance is -ve, not strongly negative though

'''
'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Approach: Creating Multidimensional Polynomial model
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


There are 6 columns as the Multidimesional Poly creates 
constant+poers of x1 and x2

and x1*x2    Refer to the theory for more details.
'''
poly_reg = PolynomialFeatures(degree=2,interaction_only=False,include_bias=True,order="C")
X_poly = poly_reg.fit_transform(X) 


'''# Creating Linear regression object'''
from sklearn.linear_model import LinearRegression
reg = LinearRegression() 

'''# Fitting X_poly and y to the model'''
reg.fit(X_poly, y)
r2_score_poly=reg.score(X_poly,y)  
print(r2_score_poly)
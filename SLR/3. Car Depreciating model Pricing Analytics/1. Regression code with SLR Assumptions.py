#Importing various modules and reading excel file from the folder
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sb
import matplotlib.pyplot as plt

# Reading the excel into a dataframe
df=pd.read_excel('price.xlsx',sheetname='depreciation')

#iloc for reading salary(dependent) and yearexperience(independent) into arrays
# Reading data into X and Y variables, if you notice X with iloc, you do not have to reshape the array lol :)
X=df.iloc[:,:-1].values
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
mmissing_counter_amit(X)

# Assumption 2 PASS

'''
**********************************
Assumption 3 of Simple Linear Regression
3. Data is free of outliers, 
If the  outliers are there take a business decision for removal @95% z=1.96 or @99.97% z=2.97

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
x=df['Year'].values
y=df['Sell_Percentage'].values
np.corrcoef(x,y)
df.cov()
sb.pairplot(df)

# Assumption 4 PASS as the relationship seems to be strongly inverse linear and corr and cov are negative

'''
**********************************
Assumption/suggestion 5 of Simple Linear Regression
5. Depending on the case/business need, data should be split into TRAIN and TEST
**********************************
'''
# Assumption 5 PASS and we are splitting data, as per the problem

'''
***************************************************************************************
***************************************************************************************
***************************************************************************************
Carrying on with Simple Linear REgression
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''

#splitting data into training and test sets
#Note that it would need 4 parameters in the same test and train sequence
#train_test_split is a function in the cross_validation module, read help for more details :)
#X and Y are the arrays that we created above using iloc, always check the dimension
#test_size=1/3 so train_size=2/3, you can give either, usually 80:20 is the norm
#random_state=0 is the seed value for replicating the results, we hardcode it for replicating the results
#The train and test would now have random values of removing any data bias and let go of data inbuilt patterns
#The name could be any for the name of the sets a,b,c,d would work, just the sequance maters
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#we import the LinearRegression class
#reg is simply a linear regressor object and LinearRegression is a class, reg is an instance of class 
#fit is a method available in LinearRegression class, this now creates a best line fit based on X and Y
#At this point the machine learning model is ready and the best fit line is in place
#this reg object cannow be called with predict method for predicting value of an array or single value
#You cannot call the predict method if the object reg is not trained using the fit method
#Y_predict are the predicted salaries and Y_test are actuals, you can compare them to see the model's performance
#creating y_predict to match with Y_test, y_predict is created with X_test 
#In summary we are trying to predict y with test x and match against test y for the accuracy of the model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_predict=reg.predict(X_test)

#We are now trying to forecast, lets say what would be price after 5.5 years
#predict is a method in LinearRegression class that helps in prediction based on trained model
#Let's say I have a car of $10,000, after 5.5 year it would be sold for $4097
ar=np.array(5.5)
print(f"The current price after depreciation is {reg.predict(ar.reshape(1,1))*10000}")

#plotting regression line and the actual observed point using the seaborn module
sb.regplot(x=df["Year"], y=df["Sell_Percentage"])

#Calculating the R square and printing it for the goodness of fit, it come out to be 98.48%
#The variation that can be explained by this model is 98.96%
r2_score=reg.score(X_train,Y_train)
print(r2_score) 


#Let's calculate the RMSE from scartch, RMSE is also a measure of the goodness of fit of a model.
#The RMSE is 6.6%
rss=((Y_test-Y_predict)**2).sum() #residual sum of squares 
mse=np.mean((Y_test-Y_predict)**2) #mean square error
print(f"Final rmse value is :{np.sqrt(mse)}")

#Importing various modules required in the model creation and graphing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sb

'''Reading the csv into a dataframe'''

df=pd.read_csv('salary.csv')

'''iloc for reading salary(dependent) and yearexperience(independent) variables into arrays
Reading data into X and Y variables, if you notice X with iloc, you do not have to reshape the array
Now the array is of 2 dimension as the scikitlearn expects the array with a 2 dimension

The 2D array helps as the algorithm is not confused whether these are 30 values/rows or 30 columns/features/variables
'''
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values

'''
**********************************
Assumption 1 of Simple Linear Regression
1. The 2 variables X and Y need to be continous numeric, NOT categorical.
In some cases the y could be ordinal.
**********************************
'''
'''
# By Inspecting the 2 variables X and Y, both are continous and Numeric
# Assumption 1 PASS

'''


'''
**********************************
Assumption 2 of Simple Linear Regression
2. Data is free of Missing values,
if missing values are there MVT(Missing Value Treatment) need to be performed.

Below function checks for the missing values :)

**********************************
'''
def mmissing_counter_amit(Y):
    if np.count_nonzero(~np.isnan(Y))==len(Y):
        print ("No missing values")
    else:
        print("Missing value count is",len(Y)-np.count_nonzero(~np.isnan(Y)), "out of" ,len(Y))
    
mmissing_counter_amit(Y)
mmissing_counter_amit(X)

''' The same summary can also be achieved using the info method
Gives the total and missing count
'''

df.info()

# Assumption 2 PASS

'''
**********************************
Assumption 3 of Simple Linear Regression
3. Data is free of outliers, 
If the  outliers are there take a business decision for removal @95% z=1.96 or @99.97% z=2.97

**********************************
'''
#np.abs is required as the z scores are negative as well, refer to z score notes
z = max(np.abs(stats.zscore(Y)))
print(z)

# Assumption 3 PASS

'''
**********************************
Assumption 4 of Simple Linear Regression
4. Linear relationship between X and Y
Correlation and covariance would give the direction and the idea about linearity
**********************************
'''
x=df['YearsExperience'].values
y=df['Salary'].values
np.corrcoef(x,y)
df.cov()
sb.pairplot(df)
# Assumption 4 PASS as the relationship seems to be strongly linear and corr and cov are positive

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
Carrying on with Simple Linear Regression
***************************************************************************************
***************************************************************************************
***************************************************************************************
'''
'''
#splitting data into training and test sets
#Note that it would need 4 parameters in the same train and test sequence
#train_test_split is a function in the model_selection module, read help for more details :)
#X and Y are the arrays that we created above using iloc, always check the dimension
#test_size=1/3 so train_size=2/3, you can give either, usually 80 (Train):20(Test) is the norm
#random_state=0 is the seed value for replicating the results, we hardcode it for replicating the results
#The train and test would now have random values, this removes any data bias and let go of data inbuilt patterns
#The name could be any for the name of the sets a,b,c,d would work, just the sequence matters (train and test)
'''

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


'''
#we import the LinearRegression class
#reg is simply a linear regressor object and LinearRegression is a class, reg is an instance of class 
#fit is a method available in LinearRegression class, this now creates a best line fit based on X and Y
#At this point the machine learning model is ready and the best fit line is in place
#this reg object can now be called with predict method for predicting value of an array or single value
#You cannot call the predict method if the object reg is not trained using the fit method
#Y_predict are the predicted salaries and Y_test are actuals, you can compare them to see the model's performance
#creating y_predict to match with Y_test, y_predict is created with X_test 
#In summary we are trying to predict y with test x and match against test y for the accuracy of the model
'''

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_predict=reg.predict(X_test)


''''We can use the attributes associates to the reg object and can print get many values'''

reg.__dict__  # All the attributes of this object are stored in the dictionary
print(reg.intercept_) # Nothing but the y intercept
print(reg.coef_) #Nothing but the coeffecient of X or the SLOPE

'''
#We are now trying to forecast, lets say what would be salary for 13 years of experince, for fun :) )
# predict is a method in LinearRegression class that helps in prediction based on trained model
#the trained model, the OLS best fitted line is with the reg object
#we are passing satic value of 13 years

'''

Y_predict_val=reg.predict(np.array(13).reshape(1,1))
print(Y_predict_val)

'''
#Implement the scatter line graph with X and Y train data
#Plotting the regression line with X and predicted Y on X train
#Basically creating line for X_train and predicted values from training data frame
# scatter is a function from plt
#in plt.plot the x is X_train and Y is predicted values of X_train or this is our y(hat), Y_train are the observed points	
'''

plt.scatter(X_train,Y_train,color='red',label="Scatter data")
plt.plot(X_train,reg.predict(X_train),c='green',label="Reg line")
#Giving title to the plot for making it look better, also giving labels to X and Y axis
plt.title('Salary and the Experience predictions for train data')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.legend()
plt.show()   

#Creating the plot for test data, very similar to the above graph
# we wish to keep the regression line as same, as this is the model line
plt.scatter(X_test,Y_test,color='red',label="Scatter data")
plt.plot(X_test,reg.predict(X_test),c='purple',label="Reg line")
#Giving title to the plot for making it look better, also giving labels to X and Y axis
plt.title('Salary and the Experience predictions for test data ')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.legend()
plt.show()  


#Model Evaluation
'''
There are three primary metrics used to evaluate linear models. 
These are: Mean absolute error (MAE), Mean squared error (MSE), or Root mean squared error (RMSE).
1.MAE: The easiest to understand. Represents average error
2. MSE: Similar to MAE but noise is exaggerated and larger errors are “punished”. 
It is harder to interpret than MAE as it’s not in base units, however, it is generally more popular.
3. RMSE: Most popular metric, similar to MSE, however, the result is square rooted 
to make it more interpretable as it’s in base units. It is recommended that RMSE 
be used as the primary metric to interpret your model.
4. R2 or R square
*******************************************************************************************;
Below, you can see how to calculate each metric. 
All of them require two lists as parameters, with one being your 
predicted values and the other being the test or true or actual values in the data'''

from sklearn import metrics

'''#1. Result for MAE (Mean Absolute Error) >> The formula is avg(|y-yhat|)'''
print(metrics.mean_absolute_error(Y_test,Y_predict))

'''#2. Result for MSE (Mean Squared Error) >> The formula is avg(|(y-yhat)**2|)'''
print(metrics.mean_squared_error(Y_test,Y_predict))

'''
#3. Let's calculate the RMSE from scartch, RMSE is also a measure of the goodness of fit of a model.
#The RMSE of $4585.41 is not bad and the model is predicting salaries to a good extent
'''
rss=((Y_test-Y_predict)**2).sum() #residual sum of squares or the SSE (Sum squared errors)..just like Variance
mse=np.mean((Y_test-Y_predict)**2) #mean square error just like the standdard deviation or sigma
print(f"Final rmse value is :{np.sqrt(mse)}")

'''#3.1 Another way for RMSE, check from above calculation, the values match'''
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_predict)))
'''
#4. Calculating the R square and printing it for the goodness of fit, it come out to be 93.8%
#The variation that can be explained by this model is 93.8%
#reg is the LinearRegression() object
'''
r2_score=reg.score(X_train,Y_train)
print(r2_score) 
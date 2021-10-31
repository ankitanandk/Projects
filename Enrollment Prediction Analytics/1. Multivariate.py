# Data Preprocessing Template for help in a standard format
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset from the working library
# Reading the first 5 variables to X 
#( X is an object as it is a mix of diff types of variables)and last to y, thats why we cannot view it, 
#you can print in console and have a look at the matrix
df = pd.read_csv('enroll.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 5].values

# Encoding categorical data is vital as these need to be encoded to numeric data with dummy variables
#The text would not go well with the mathematical equations
#Labelencoder is a class in sklearn, it encodes data to 0,1 and 2, this may be misleading as we are giving weight to Nominal data
#To sort out the above issue we use dummy variables, OnehOtEncoder is used for the same
#Onehot encoder works on the encoded data, so after Labelencoer has finised the job
# Encoding the Independent Variable "State"
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#Encoding the third column, which is state :), the LabelEncoder changes the text to number
#Till this point we would have one variable with values 0,1 and 2 as we have 3 states, the data type is object
#Print in console and see the data 
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# This would convert the values in 0,1,2 series and flag the data for a particular value in 3 columns
# Onehot encoder cannot directly work on Text data and it needs numbers created by LabelEncoder
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap, though the model takes care of this, we may need to consult the documentation
#However we are dropping 1st dummy variable, remember always have n-1 dummy variables to avaoid multicollinearity
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set, in the same 80:20 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Fitting Multiple Linear Regression to the Training set of X and y :)
# This is still a linear regression model, just with many Xs
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#Predicting the test results, now just compare the predicted results with y_test
# The y_test is the actual values and the y_pred are predicted ones, y_test is used to comapre with y_pred for residual Analysis
# Honestly at this stage,we have a model that can predict profit from all the available variables
# The Machine has learned and now we can make a prediction for the new set of data
y_pred=reg.predict(X_test)

#The R square seems to be very good for this model, the model explains 94.63% 
r2_score=reg.score(X_train,y_train)
print(r2_score)

# Building the optimal multivariate model using Backward Elimination
# We are trying to find the best combinations of x vars that explain y, how these x interact and lead to y
# We are adding ones to X to account for b0 or 1 as the coefficient of b0,1b0
# np.ones for the identity matrix and astype for converting to int, remember from the numpy learning :)
# we are adding ones in begining to X and thus array is np.ones
# We are doing this as the statsmodels library does not account for b0, the LinearRegression does :)
#Thats why we did not created identity matrix in Simple Linear Regression
#If you dont add identity in start the eq would be =(b1x1+b2x2)
#since we have given np.ones first the column comes first and then the rest of data
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((29, 1)).astype(int), values = X, axis = 1)

# Creating optimal features of X, its just a copy, we call it as Optimal X
# New array with all columns, or you could have simply said X_opt=X[:], this would help in elimination
# This is the future state that we want our X to look like
X_opt = X[:, [0, 1, 2, 4, 5,6]]
# New object from the class OLS(Ordinary Least Squares), endog is endogenous or the Y
#And the exog is the indep or the X serie s, and endog is Y :)
#Read the intercept is not included in the model, thats why we have to add it using np.ones :)
# Ctrl I on the read the description for OLS
#.fit is the fit method for fitting OLS for X and y
# Till this point we have fit full model with all predictors
# The summary() function is helping us to see the p values and creates the summary table

#From here the backward elimination starts
# The p value was .804, way higher than .05 :)
# We keep on running till the time we are left with the significant few
X_opt = X[:, [0, 1, 2,3 ,4, 5,6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,1,3,4,5,6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5,6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [3,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


#The final model, thus the only significant variable by the backward elimination is unemployed rate
X_opt = X[:, [3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()









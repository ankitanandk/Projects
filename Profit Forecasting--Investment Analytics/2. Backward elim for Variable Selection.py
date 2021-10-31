'''
# Building the optimal multivariate model using Backward Elimination
# We are trying to find the best combinations of x vars that explain y, how these x interact and lead to y
# We are adding ones to X to account for b0 or 1 as the coefficient of b0,1b0
# np.ones for the identity matrix and astype for converting to int, remember from the numpy learning :)
# we are adding ones in begining to X and thus array is np.ones
# We are doing this as the statsmodels library does not account for b0, the LinearRegression does :)
#Thats why we did not created identity matrix in Simple Linear Regression
#If you dont add identity in start the eq would be =(b1x1+b2x2)
#since we have given np.ones first the column comes first and then the rest of data
'''
import numpy as np
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
'''
# Creating optimal features of X, its just a copy, we call it as Optimal X
# New array with all columns, or you could have simply said X_opt=X[:], this would help in elimination of predictors
# This is the future state that we want our X to look like
'''
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
x_opt = pd.DataFrame(X, columns=['constant','State 1','State 2','R&D','Administration','Marketing Spend'])
type(x_opt)
'''
# New object from the class OLS(Ordinary Least Squares), endog is endogenous or the Y
#And the exog is the indep or the X series
#Read the intercept is not included in the model, 
thats why we have to add it using np.ones :)

You can even add the constant using the 

statsmodels.tools.add_constant.



# Ctrl I on the read the description for OLS
#.fit is the fit method for fitting OLS for X and y
# Till this point we have fit full model with all predictors

'''
import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()


'''
# The below summary() function is helping us to see the p values and creates the summary table
'''
regressor_OLS.summary()

'''
#From here the backward elimination starts, we have now removed predictor 2 
# The p value was .990, way higher than .05 :)
# We keep on running till the time we are left with the significant few
'''
#X_opt = X[:, [0, 1, 3, 4, 5]]
x_opt = x_opt.drop('State 2',axis=1)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#X_opt = X[:, [0, 3, 4, 5]]
x_opt = x_opt.drop('State 1',axis=1)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#X_opt = X[:, [0, 3, 5]]
x_opt = x_opt.drop('Administration',axis=1)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
'''
#Finally this tells that R&D spend is the most significant and critical variable(X) :)
'''
#X_opt = X[:, [0, 3]]
x_opt = x_opt.drop('Marketing Spend',axis=1)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


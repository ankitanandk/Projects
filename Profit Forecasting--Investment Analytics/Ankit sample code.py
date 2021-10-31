import pandas as pd

df =pd.read_csv('50_startups.csv')
x = df.iloc[:,:4]
y = df.iloc[:,-1].values

x=pd.get_dummies(x,drop_first='True')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train,y_train)
y_pred = linreg.predict(x_test)

from sklearn.metrics import r2_score

r2_s = r2_score(y_test,y_pred)

import numpy as np

x = np.append(np.ones((50,1)).astype(int),values = x, axis =1)

x_opt = pd.DataFrame(x, columns=['constant','R&D','Operation','Marketing','State 1','State 2'])

import statsmodels.api as sm

regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()

print(regressor_OLS.summary())

x_opt = x_opt.drop('State 1',axis=1)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = x_opt.drop('State 2',axis=1)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = x_opt.drop('Operation',axis=1)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())


x_opt = x_opt.drop('Marketing',axis=1)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
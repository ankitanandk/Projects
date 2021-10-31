#Importing various modules and reading the new csv, new data there are 10 rows and extrapolated exp of 11 and 13 have come
#we just have X (exp) and need to predict the salaries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_new=pd.read_csv('salary_new.csv')

# Using the iloc for create a array, remember no need of reshaping it :)
X_test_new=df_new.iloc[:].values

#predicting new salary by using the earlier trained model
#remember the reg object should be alive in the session, you cannot have a brand new sesion and use below code.
#reg is storing the best fitted line OLS model and we can use the same for predicting salaries
Y_predict_sal_new=reg.predict(X_test_new)

#look for 11 yr exp sal is 129621 and for 13 yrs it is 148313, same as that of excel predcited values 
print(X_test_new,Y_predict_sal_new)
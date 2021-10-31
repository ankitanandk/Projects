'''
#Importing various modules and reading the new csv, new data there are 10 rows
'''

import pandas as pd
df_new=pd.read_csv('Social_Network_Ads_new.csv')

'''
# Using the iloc for create a array, remember no need of reshaping it :)
'''
X1_test_new = df_new.iloc[:, [2, 3]].values

'''scaling on new x'''

from sklearn.preprocessing import StandardScaler
sc_new = StandardScaler()
X1_test_new = sc.fit_transform(X1_test_new)

X2_new=df_new[['Gender']]

''' The Dummy variables should be created for the categorical variable'''
from sklearn.preprocessing import LabelEncoder
labelencoder_X2 = LabelEncoder()
X2_new=X2_new.apply(LabelEncoder().fit_transform)

X1_test_new=np.concatenate((X1_test_new,X2_new),axis = 1)


del X2_new

'''
#predicting new salary by using the earlier trained model
#remember the reg object should be alive in the session, you cannot have a brand new sesion and use below code.
#reg is storing the best fitted line OLS model and we can use the same for predicting salaries
'''
Y_predict_new=classifier.predict(X1_test_new)


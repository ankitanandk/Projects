'''
#*************************************************************************

#Reading the new data and applying the model for predicting the actual days

#*************************************************************************

'''
df_new10 = pd.read_csv('allocations_new10.csv')
X1_new10 = df_new10.iloc[:, 0:3]
X2_new10=  df_new10.iloc[:, 3:4]
# Encoding the Independent Variable Analyst Level, Complexity and data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X1_new10=X1_new10.apply(LabelEncoder().fit_transform)

#Adding the encoded data and the Height column and concatenating the X1 and X2 arrays
X_new10=pd.concat((X1_new10,X2_new10),axis = 1)
X_new10=X_new10.values
#Deleting the intermediate frames, these would no longer be used in the subsequent code
del X1_new10,X2_new10
y_new10= regressor.predict(X_new10)

'''Validating the RF model by the score on train and the test or the OOB score
The score of the Defaut Random Forest model, The score is 95.28% for 300 trees
'''
print(regressor.score(X,y))


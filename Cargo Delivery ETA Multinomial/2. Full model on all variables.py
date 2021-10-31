'''Logistic Regression for predicting the event, this is best used as a Binary classifier
However lets try to predict the multiple categories using thos algorithm
Importing the libraries for the downstream processing'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Importing the dataset and reading age and salary to X, let's assume we are just looking for this only
for Simplicity lets take 2 variables in the X for plotting in 2 Dimension
'''
df = pd.read_csv('delivery.csv')
X1 = df.iloc[:, [2, 3]]
X2=df[['Type']]

from sklearn.preprocessing import LabelEncoder
labelencoder_X2 = LabelEncoder()
X2=X2.apply(LabelEncoder().fit_transform)

X=np.concatenate((X2,X1),axis = 1)
del X1,X2
y = df.iloc[:, 4]

'''
Splitting the dataset into the Training set and Test set
'''
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
Feature Scaling needs to be performed as the values need to be normalized
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


'''
# Fitting Logistic Regression to the Training set
The OVR ( or the One over rest is leveraged for the calculation for evry class)
'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0,multi_class='multinomial')

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

'''
# Predicting the Test set results
'''
y_pred = classifier.predict(X_test)

'''
#**********************Evaluating/Validating the model on Ytest ad Ypred***********************************

# Making the Confusion Matrix
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
The score could be validated by adding the diagonal values
The score is 71 or 71% accuracy of the logistic model
'''

classifier.score(X_test, y_test)






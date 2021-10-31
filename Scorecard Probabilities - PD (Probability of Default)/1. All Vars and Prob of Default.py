'''Logistic Regression for predicting the event, this is best used as a Binary classifier
Importing the libraries for the downstream processing

Let's try and create a model that would give a score or a %age of every customer rather than the 0 or 1

This %age can now be used as a score for prediction of Prob of Default
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''Importing the dataset and reading age and salary to X, let's assume we are just looking for this only
for cimplicity lets take 2 variables in the X for plotting in 2 Dimension
Age and Salary as X

The Defaulters are who default on the payments and y=1, if y=0 it is not a defaulter
'''
df = pd.read_csv('Defaulters.csv')
X1 = df.iloc[:, [2, 3]]
X2=df[['Gender']]

from sklearn.preprocessing import LabelEncoder
labelencoder_X2 = LabelEncoder()
X2=X2.apply(LabelEncoder().fit_transform)

X=np.concatenate((X1,X2),axis = 1)
del X1,X2
y = df.iloc[:, 4].values

'''Splitting the dataset into the Training set and Test set'''
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''Feature Scaling needs to be performed as the values need to be normalized
why are we not scaling y?
Y is 0 and 1 thus no need for scaling
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

'''
Predict: Predicts the probability and give output in 0 and 1 format


Predict_proba: Give the array output and shows the probability of 0 and 1.
This now predicts the probability of 0 or "No"  and 1 or "YES" based on the business problem.

This can also be leveraged for a scorecard creation and other score problems

Just taking the probability of Default or the probability of 1
Just keeping it simple, this is the Probablity of Default in terms of absolute 1 as per the given data
This can be converted to any score for a scorecard derivation
'''

probs = classifier.predict_proba(X_test)
probs1 = classifier.predict(X_test) # Just to compare

'''
Just keeping one set , the other set could be easily calculated for the data
As they both are complimentary: The sum should be 1 or 100%
'''
scorecard = probs[:,1:]*100

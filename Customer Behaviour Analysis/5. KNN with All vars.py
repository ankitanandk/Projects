'''K-Nearest Neighbors (K-NN) is one of the most powerful and widely used classifier in ML
We would try and see if we get a better confusion matrix with KNN as compared to Logistic Regression
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
# Importing the dataset, reading all variables, we wont be able to plot now
'''
df = pd.read_csv('Social_Network_Ads.csv')
X1 = df.iloc[:, [2, 3]].values
X2=df[['Gender']]

from sklearn.preprocessing import LabelEncoder
labelencoder_X2 = LabelEncoder()
X2=X2.apply(LabelEncoder().fit_transform)

X=np.concatenate((X1,X2),axis = 1)
del X1,X2
y = df.iloc[:, 4].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling for normalizing the salary values, else they would impact the euclidean distance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''Fitting K-NN to the Training set, we import the library and create the object
Metric is 'minkowski' and p=2 is for eucledian distance, refer documentation please :)

We would later try and optimize the value of K(# of neighbours) and p=1 or p=2

'''
'''
The value of p is explained below.
p : integer, optional (default = 2)
Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), 
and euclidean_distance (l2) for p = 2.
 For arbitrary p, minkowski_distance (l_p) is used.
 '''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#****Evaluating the model based on confusion matrix score, sensitivity and specificity*************

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Validating the KNN Model using the score and it is 93%, far better than the logistic Regression score of 89%
# How can you validate the 93% score manually from the confusion matrix
classifier.score(X_test, y_test)

# Calculating the TP,TN,FP and FN ,,, these are used for calculating specificity and sensitivity
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

#Classification Accuracy: Overall, how often is the classifier correct? Does it match as above?

print((TP + TN) / float(TP + TN + FP + FN))

#Classification Error: Overall, how often is the classifier incorrect? Also known as "Misclassification Rate"
# Why it is 7%
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)

'''
#*******Sensitivity or the Recall This basically share how good the model is identifying the 1's better
as compared to the 0's 
'''
sensitivity = TP / float(FN + TP)
print(sensitivity)

#*******Specificity**********
specificity = TN / (TN + FP)
print(specificity)

#***********We cannot visualize the results as now we have more than 2 Dimension ************

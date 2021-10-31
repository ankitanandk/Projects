'''K-Nearest Neighbors (K-NN) for the survey score classification with multiple y values 6-10 (5 distinct values)

Case of Ordinal Classification as the categories are in a order and rank, yet the arithmetic calculation can not be performed :)

'''
'''Importing the libraries for the downstream processing'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Importing the dataset surevy with multiple y values'''

dataset = pd.read_csv('Survey.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

'''
# Splitting the dataset into the Training set and Test set
'''
from sklearn.model_selection  import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
# Feature Scaling for the variables to reduce the impacton geometric distances
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
# Fitting K-NN to the Training set, we import the library and create the object
# Metric is 'minkowski' and p=2 is for eucledian distance, refer documentation please :)
'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

''' Predicting the Test set results'''
y_pred = classifier.predict(X_test)

'''
#************************************Evaluating the model***********************************

# Making the Confusion Matrix
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
#Validating the KNN Model using the score and it is 61%. 
Thus it is able to classify the values in 5 categories with 61% accuracy

Look at the Diagonal and see that 16+5+9+19+12 are the correct classification of the customers

Rest all is the Error or the miss classification.

'''
classifier.score(X_test, y_test)

'''
#******************************************************************************************
# Visualising the Test set results in the 2 dimension for all the five categories

All the alien colurs in the map are the miss classifications.
'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue','yellow','brown')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue','yellow','brown'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
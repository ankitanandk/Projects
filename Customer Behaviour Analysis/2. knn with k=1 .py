''' K-Nearest Neighbors (K-NN) with k=1 the boundaries would not be smooth, 
# logistic regression is a linear classifier and thus there was a line separating the 0 and 1
# KNN would make contours or pockets for the best fit of 0 and 1 based on the train data, as it is a non linear classifier

This fitting may not be ideal for the test data and we may see more validation error rather than the train error.

To see the diffrence in Train and Validation error, refer to the notes :)
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set, we import the library and create the object
# Metric is 'minkowski' and p=2 is for eucledian distance, refer documentation please :)
# With knn as 1 or 1knn, we are shoosing the nearest neighbours and no longer taking votes from neghbours
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#************************************Evaluating the model***********************************

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Validating the KNN Model using the score and it is 87%, less than the logistic score of 89% and 93% of knn=5
# How can you validate the 87% score manually from the confusion matrix
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

#*******Sensitivity**********

sensitivity = TP / float(FN + TP)
print(sensitivity)

#*******Specificity**********
specificity = TN / (TN + FP)
print(specificity)

#******************************************************************************************

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
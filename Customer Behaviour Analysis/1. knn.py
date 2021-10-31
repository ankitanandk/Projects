'''K-Nearest Neighbors (K-NN) is one of the most powerful and widely used classifier in ML
We would try and see if we get a better confusion matrix with KNN as compared to Logistic Regression
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Importing the dataset, just reading 2 variables for easy and clear plotting in 2D'''
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

'''Splitting the dataset into the Training set and Test set with 75% and 25% ratios'''
from sklearn.model_selection  import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''Feature Scaling for normalizing the salary values, else they would impact the euclidean distance
the salary values would impact the distnace to a greater extent as compared to Age

*************************************
fit_transform means to do some calculation and then do transformation 
(say calculating the means of columns from some data and then replacing the missing values).
 So for training set, you need to both calculate and do transformation.
 
But for testing set, Machine learning applies prediction based on what was 
learned during the training set and so it doesn't need to calculate, it just performs the transformation.

Example for more clairt :)

scaler = StandardScaler()
scaler.fit(X_train)  # get the 2 parameters from data (**μ and σ**) as we need the Z value for scaling
scaler.transform(X_train) # apply scale with given parameters

scaler.transform(X_test) # apply scale with training parameters on the testing data

It is just a shortcutm:)
and you can use fit_transform(X_train) for shortcut rather than fit(X_train) => transform(X_train)

*************************************
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

'''Fitting K-NN to the Training set, we import the library and create the object
Metric is 'minkowski' and p=2 is for eucledian distance, refer documentation please :)

for p=1 the minkowski transforms to the Manhattan distnace

We would later try and optimize the value of K(# of neighbours) and p=1 or p=2 using the gridsearchCV

'''

'''
The value of p is explained below.
p : integer, optional (default = 2)
Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), 
and euclidean_distance (l2) for p = 2.
 For arbitrary p, minkowski_distance (l_p) is used. For example for p=3 the actual values get calculated using the formula
 '''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



'''****Evaluating the model based on confusion matrix score, sensitivity and specificity*************'''

''' Making the Confusion Matrix for calculating the sensitivity, specificity, accuracy and the error'''
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

#******************************************************************************************

'''Visualising the Training set results'''
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

'''Visualising the Test set results, KNN is not a linear classifier, this is the reason for contours
 Logistic Regression was a linear classifier and thus a straight line as separating boundary, looks like a map
 the accuracy is better than the Logistic Regression
 '''
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
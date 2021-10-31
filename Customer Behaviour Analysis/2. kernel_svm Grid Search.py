# Kernel SVM with rbf or radial basis function

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

# Fitting Kernel SVM to the Training set, lets try linear kernel and see with Gridsearch for best parameters
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Grid Search and setting the parameters, let's try for the best values of kernel ,c and gamma
# Parameter Grid

'''
C: The Penalty Parameter

High C >>The classifier is less tolerant to the missclassification. Default is C=1 ****

With High C the error would be less; but the bias would shoot up

The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. 
For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a 
better job of getting all the training points classified correctly. Conversely, a very small value of C will
 cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies
 more points. For very tiny values of C, you should get misclassified examples,
 often even if your training data is linearly separable.
 
 The C parameter trades off misclassification of training examples against simplicity of the decision surface.
 A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly
 by giving the model freedom to select more samples as support vectors.
 
************************************************************************************************************;
 
gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
Current default is 'auto' which uses 1 / n_features
Gamma is the parameter of a Gaussian Kernel (to handle non-linear classification). At high Gamma the
classifier becomes more curvy and tends to adapt to all data points.
'''

param_grid = {'kernel':['linear','rbf','poly','sigmoid'],'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
 
# Make grid search classifier with cross validatiing 3 times, so every data is trained 2 times :)
from sklearn.model_selection import GridSearchCV
clf_grid = GridSearchCV(SVC(), param_grid, cv=3)
 
# Train the classifier
clf_grid.fit(X_train, y_train)
 
# clf = grid.best_estimator_()
# what are the best parameters, 
# lets try and get the confusion matrix, sensitivity and specificity with these parameters
print("Best Parameters in Kernel SVM are: ", clf_grid.best_params_)

#****Evaluating the model based on best parameters*************

# Fitting Kernel SVM to the Training set with the best parameters
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0,C=1,gamma=1)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


#****Evaluating the model based on confusion matrix score, sensitivity and specificity*************

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''The best score with rbf and c=1 and gamma=1 is 93%'''
classifier.score(X_test, y_test)

# Calculating the TP,TN,FP and FN ,,, these are used for calculating specificity and sensitivity
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

#Classification Accuracy: Overall, how often is the classifier correct? Does it match as above?

print((TP + TN) / float(TP + TN + FP + FN))

#Classification Error: Overall, how often is the classifier incorrect? Also known as "Misclassification Rate"
# Why it is 10%
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)

#*******Sensitivity**********

sensitivity = TP / float(FN + TP)
print(sensitivity)

#*******Specificity**********
specificity = TN / (TN + FP)
print(specificity)

#****************************************************************************************************
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
plt.title('Kernel SVM with rbf (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

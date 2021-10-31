'''# K-Nearest Neighbors (K-NN) with Grid search for the optimization of the model
# To achieve the model parameters from the hyperparemeters in KNN.
# Model parameters is the best combination of hyperparameters at which model performs the best
'''

'''Importing the library for further use in the model code'''
import pandas as pd

'''Importing the dataset'''
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

'''Splitting the dataset into the Training set and Test set'''

from sklearn.model_selection  import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''Feature Scaling'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
#**Evaluating the model using the Grid Search for best value of k and choice of distance on train data***
# We are training the data for knn=3, lets see if the grid search is able to identify the best knn for train data
and the optimal p value. IF p=1 and p=2 then it would be either a Euclidean or a Manhattan distance
'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[3,4,5,6,7,8,9,10],'metric':['minkowski'],'p':[1,2,3,4,5,6,7,8,9,10]}

model = GridSearchCV(classifier, params,cv=3)
gs_results=model.fit(X_train,y_train)
gs_results.best_params_
gs_results.best_score_

'''
#*******Evaluating the model with the best model parameters from the grid search hyper parameters********

# Predicting the Test set results with the above recommendation from Grid search of k=5 and dist=minkowski and p=5

'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

'''
Validating the KNN Model using the score and it is 93%, far better than the logistic score of 89%
The score has not improved based on the last code; however this is the best combinationof hyperparameter tuning and CV=3
'''
classifier.score(X_test, y_test)

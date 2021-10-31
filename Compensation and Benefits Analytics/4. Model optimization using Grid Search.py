# Random Forest Regression: The Enseble method for Regression

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#The default trees or the estimator is 10, we are explicitly calling that in below code
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0,oob_score=True)

#Fit method for the fittment

regressor.fit(X, y)


'''Validating the RF model by the score on train and the test or the OOB score
The score of the Defaut Random Forest model, the score of 97.04% may not be the actual representation'''
print(regressor.score(X, y))

'''Always Validate the RF model usingthe OOB score, which is Out Of Bag score
This is the score that tells how well the model fits on OOB observations
Look at the OOB score, it is around 68% and the flag should be set to True in regressor'''
print(regressor.oob_score_)

#**Evaluating the model using the Grid Search for best value of tree and leaf size***

from sklearn.model_selection import GridSearchCV
params = {'n_estimators':[5,10,25,50,100,150,200,250,300],
          'min_samples_leaf':[1,2,3]}

# we are also optimizing the model by CROSS VALIDATION, CV=3 is usually used in gridsearch
model = GridSearchCV(regressor, params,cv=3)
rf_results=model.fit(X,y)
rf_results.best_params_

'''To capture the entire GRID in a dictionary for easy understanding
The ranking happens on the basis of mean_test_score.. the tree having max value gets ranks 1
The mean_test_score is avg of mean_test_score0-mean_test_score2 as we have used CV=3

look for second tree with leaf=1 and Tree=25 that has highesht mean_test_score and thus best tree
'''
cv_res=rf_results.cv_results_





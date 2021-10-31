'''
Just create the regressor by fitting on any no of trees

'''
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 2, random_state = 0,oob_score=True)
regressor.fit(X, y)



'''
**Evaluating the model using the Grid Search for best value of tree and leaf size***
and depth and you can pass all the possible values of any parameter
'''

from sklearn.model_selection import GridSearchCV
params = {'n_estimators':[5,10,25,50,100,150,200,250,300],
          'min_samples_leaf':[1,2,3],'max_depth' :[i for i in range(1,11)]}

'''
# we are also optimizing the model by CROSS VALIDATION, CV=3 is usually used in gridsearch
CV=3 is used when we wish to cross validate the data 3 times
'''
model = GridSearchCV(regressor, params,cv=3)
rf_results=model.fit(X,y)



rf_results.best_params_





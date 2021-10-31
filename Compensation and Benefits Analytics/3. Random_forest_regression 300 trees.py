# Random Forest Regression with 300 tress and let's see how the ensemble helps the RF

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Random Forest Regression to the dataset
# Creating an object, Fit and then Predict, Remember these three steps for all models :)
# N_estimators means how many Trees and we would take the average of all Trees, Trees in your Forest
# Default values is 10, Random_state is again the seed for the random number
# Lets see this with 300 Trees, The sal comes out to be very very close to 160 k :) :)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0,oob_score=True)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(np.array([[6.5]]))

# Visualising the Random Forest Regression results (higher resolution), 
#This high resolution is required for Random Forest as well, remember it is a Ensemble method of many models
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''Validating the RF model by the score on train and the test or the OOB score
The score of the Defaut Random Forest model, The score is 95.28% for 300 trees
'''
print(regressor.score(X, y))

'''Always Validate the RF model usingthe OOB score, which is Out Of Bag score
This is the score that tells how well the model fits on OOB observations'''
print(regressor.oob_score_)
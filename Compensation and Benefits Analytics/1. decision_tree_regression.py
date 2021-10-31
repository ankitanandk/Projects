# Decision Tree Regression using a single Tree

# Importing the libraries for downstream codes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and creating X and y
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting Decision Tree Regression to the dataset, creating the object from the class and using fit method
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result for an experience of 6.5 years
y_pred = regressor.predict(np.array([[6.5]]))


'''Visualising the data
Do not see this in low resolution, else the categories woul be lost :)
The plot shows the continuity; however the tree should be non continous model and we should see steps.

For that let's increase the resolution and create 100 datapoints in the below code
'''
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


'''Visualising the Decision Tree Regression results (higher resolution)
Do not see this in low resolution, else the categories woul be lost :)
These 100 values of X would give a staircase picture and thats a better explanantion of the tree.
The values in a stair is the average of all data points in that interval
'''
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression in HIGH resolution)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''Let's try and see the inverted tree that was used by the model using mse as the parameter for split
# import export_graphviz   
# export the decision tree to a enrollment.dot file 
# for visualizing the plot easily anywhere ...http://www.webgraphviz.com/ 
jst push the data and Tree would be created'''

from sklearn.tree import export_graphviz  
export_graphviz(regressor, out_file ='Salary_predictions.dot', 
               feature_names =['Position'])  
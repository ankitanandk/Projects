# Decision Tree Regression from CART

# Importing the libraries for the regression analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and reading X and Y 
df = pd.read_csv('petrol_consumption.csv')
X = df.iloc[:, 0:4].values
y = df.iloc[:, -1].values

#Splitting the data in test and train for the optimal ouput
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

# Fitting Decision Tree Regression to the dataset, creating the object from the class.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the results as per the test data and y_pred can now be compared to y_test
# This comparison would give us the delta or te residuals of the model
y_pred = regressor.predict(X_test)  

# Let's try and see theinverted tree that was used by the model using mse as the parameter for split
# import export_graphviz   
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere ...
# http://www.webgraphviz.com/ jst push the data and Tree would be created
from sklearn.tree import export_graphviz  
export_graphviz(regressor, out_file ='consumption.dot', 
               feature_names =['Petrol_tax', 'Average_income', 'Paved_Highways',
       'Population_Driver_licence(%)'])  
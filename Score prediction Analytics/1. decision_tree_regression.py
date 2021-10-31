'''
# Decision Tree Regression from CART ( we are just using the Regression Tree)

# Importing the libraries for the regression analysis
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
# Importing the dataset and reading X and Y  as the dataframe and not as an array
'''
df = pd.read_csv('student_scores.csv')
X = df.iloc[:, 0:3]
y = df.iloc[:, -1]

'''
#Splitting the data in test and train for the optimal ouput
'''
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

'''
# Fitting Decision Tree Regression to the dataset, creating the object from the class.

Look at the criteria used for split 

mse: mean square error or the variance or the Standard deviation or sigma
'''
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

'''
# Predicting the results as per the test data and y_pred can now be compared to y_test
# This comparison would give us the delta or te residuals of the model
'''
y_pred = regressor.predict(X_test)  

'''
#The r score of the CART model is usually high as the data tends to overfit, 
# Thats why we tend to limit the branch size
'''
score=regressor.score(X_test,y_test)

'''
#Let's try and predict the scoreof 1 new joiners using the CART model

'''
y_pred_sample=regressor.predict(np.array([45000,1,9]).reshape(1,3))

'''
# Let's try and see theinverted tree that was used by the model using mse as the parameter for split
# import export_graphviz   
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere ...
# http://www.webgraphviz.com/ jst push the data and Tree would be created
'''
from sklearn.tree import export_graphviz  
export_graphviz(regressor, out_file ='score.dot', feature_names =X.columns, 
                filled=True,node_ids=True, rotate=True)  



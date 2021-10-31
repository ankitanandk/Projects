# Decision Tree Regression from CART, This is a classification problem as the response variable isnot continous
# The intent of this code is to see compare the mse scores of the decision tree and the manual calculation

# Importing the libraries for the regression analysis
import numpy as np
import pandas as pd

# Importing the dataset and reading X and Y and doing the label encoding
df = pd.read_csv('enroll.csv')
X1=df[['Gender','Class']]
X2=df[['Height']].values
y = df.iloc[:, -1].values

# Encoding the Independent Variable "Gender" and "Class"
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X1=X1.apply(LabelEncoder().fit_transform)

#Adding the encoded data and the Height column and concatenating the X1 and X2 arrays
X=np.concatenate((X1,X2),axis = 1)

#Deleting the intermediate frames, these would no longer be used in the subsequent code
del X1,X2

# Fitting Decision Tree Regression to the dataset, creating the object from the class.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


'''Let's try and see theinverted tree that was used by the model using mse as the parameter for split
import export_graphviz    export the decision tree to a enrollment.dot file 
for visualizing the plot easily anywhere ...http://www.webgraphviz.com/ jst push the data and Tree would be created
'''
from sklearn.tree import export_graphviz  
export_graphviz(regressor, out_file ='tree.dot', 
               feature_names =['Gender', 'Class','Height'])  

import pydot
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
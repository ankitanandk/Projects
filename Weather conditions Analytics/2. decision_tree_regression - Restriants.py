# Decision Tree Regression from CART, This is a classification problem as the response variable isnot continous
# The intent of this code is to see compare the mse scores of the decision tree and the manual calculation

# Importing the libraries for the regression analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and reading X and Y and doing the label encoding
df = pd.read_excel('weather.xlsx')
X=df.iloc[:,0:-1]
y = df.iloc[:, -1].values

# Encoding the Independent Variable "Gender" and "Class"
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X=X.apply(LabelEncoder().fit_transform)

# Fitting Decision Tree Regression to the dataset, creating the object from the class.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0, min_samples_split=2, min_samples_leaf=1)
regressor.fit(X, y)
r2score=regressor.score(X,y)

# Let's try and see theinverted tree that was used by the model using mse as the parameter for split
# import export_graphviz   
# export the decision tree to a enrollment.dot file 
# for visualizing the plot easily anywhere ...http://www.webgraphviz.com/ jst push the data and Tree would be created
from sklearn.tree import export_graphviz  
export_graphviz(regressor, out_file ='weather.dot', 
               feature_names =['outlook', 'temp', 'humidity', 'windy'])  

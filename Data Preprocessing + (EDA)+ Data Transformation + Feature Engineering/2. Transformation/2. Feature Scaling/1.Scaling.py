#1.Data processing, below are the libraries we that we would need for all ML sessions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset, remember to set a working directory in Spyder
#Separating data in X(country, age and sal) and Y(Purchased)
#For printing you can use the console as well
df = pd.read_csv('Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values


# Taking care of missing data, Imputer is a class from the sklearn library
# The missing data in above df would be stored as  'nann'
from sklearn.preprocessing import Imputer
#imputer, we are creating an object from the Imputer class
#Mean is also the default stratgey in Imputer
#axis =0 is the mean of columns and 1 is the mean of rows
#Ctrl +I for inspect and you can check the documentation in Python
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
type(imputer)
print(imputer)
#Fitting Imputer to our X, we are doing this for all rows and just columns 1 and 2, upperbound 3 is excluded :)
impuer=imputer.fit(X[:,1:3])
#Applying method transform for pushing the mean values
X[:, 1:3] = imputer.transform(X[:, 1:3] )

# Encoding categorical data is vital as these need to be encoded to numeric data, the model doesnot understand this data
#Labelencoder is a class in sklearn, encoding the categorical variables to to 0,1 and 2
#This may be misleading as it is Ordinal data and thats not how categorical variable should be stored
#To sort out the above issue we use dummy variables, OneHotEncoder is used for the same, to create a flag for all values
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Encoding the Independent Variable and creating dummy values
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable, this doesnot need ONHOtEncoder as it is the Y variable and python knows it
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set, the norm is 80:20
# Random state is the seed used by the random generator
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling, used for scaling the variables
# For the X_train we fit and transform simultaneoulsy 
#Note we are scaling the Dummy variables as well, we loose the interpretation though, and it is case based
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# We just transform the X test, fitting is not required as we would use test for prediction
X_test = sc_X.transform(X_test)

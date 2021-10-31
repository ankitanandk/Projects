'''Data Preprocessing Template for help in a standard format
Importing the libraries, at  this stage pandas is only required'''
import pandas as pd

'''Importing the dataset from the working library
Reading the first 4 variables to X's
( X is an object as it is a mix of diff types of variables, look attype)and last to y, 
thats why we cannot view it, you can print in console and have a look at the matrix 
Use the print command print(X) for the same in console
'''
df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

'''Encoding categorical data is vital as these need to be encoded to numeric data with dummy variables
The text would not go well with the mathematical equations and calculations
Labelencoder is a class in sklearn, it encodes data to 0,1 and 2, 
this may be misleading as we are giving weights to Nominal data with a factor, like 2X.
To sort out the above issue we use dummy variables, OnehOtEncoder is used for the same
Onehot encoder works on the encoded data, so after Labelencoer has finised the job
Encoding the Independent Variable "State" using onehotencoder'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
'''Encoding the third column, which is state :), the LabelEncoder changes the text to number
Till this point we would have one variable with values 0,1 and 2 as we have 3 states, the data type is object
Print in console and see the data
The 0,1 and 2 comes by sorting the state variable ascending: Cal: 0, Fl: 1 and NY:2
The Analogy can be drawn to "if else if" for creating the label encoding  
'''
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

'''Printing the object, as it cant be opened directly'''
print(X)

'''This would convert the values in 0,1,2 series and flag the data for a particular value in 3 columns
Onehot encoder cannot directly work on Text data and it needs numbers created by LabelEncoder
The 3 dummy variables are now ready

[3] basically means to work on the 3rd variable for encoding the distinct values to that many dummy variables
'''
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

'''Avoiding the Dummy Variable Trap, though the model takes care of this, we may need to consult the documentation
However we are dropping 1st dummy variable, remember always have n-1 dummy variables to avoid multicollinearity'''
X = X[:, 1:]

'''We have another method if you find that your spyder is newer version and this method is depreciated
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)
x = x[:, 1:]
'''

'''M3
x=pd.get_dummies(x,drop_first='True')
'''

# Splitting the dataset into the Training set and Test set, in the same 80:20 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
'''Fitting Multiple Linear Regression to the Training set of X and y :)
This is still a linear regression model, just with many Xs
This is the ALL IN approach where we are not eliminating any variable, pushing all to the model
'''
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

print(reg.__dict__)

'''Predicting the test results, now just compare the predicted results with y_test
The y_test is the actual values and the y_pred are predicted ones, y_test is used to comapre with y_pred for residual Analysis
Honestly at this stage,we have a model that can predict profit from amount spent on R&D ,Admin & Marketing
The Machine has learned and now we can make a prediction for the new set of data'''
y_pred=reg.predict(X_test)

'''
Calculating The R2 for the model on y_test and y_pred
It comes out to be 93.4% that is very very decent
'''
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

















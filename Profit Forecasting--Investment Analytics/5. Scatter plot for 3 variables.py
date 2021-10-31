# Tryingto plot the independent variables on a plot to visualize the plane
import matplotlib.pyplot as plt
import pandas as pd

#Though you would need 3D for viewing more than 2 variables, we are tryingto see 3 variables on a scatter

df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
fig, ax = plt.subplots()
x1 = X[:,2]
x2 = X[:,3]
x3 = X[:,4]
scat=ax.scatter(x1, x2,c=x3, marker="x")
fig.colorbar(scat)
plt.show()

    
















'''Logistic Regression for predicting the event, this is best used as a Binary classifier
However lets try to predict the multiple categories using thos algorithm
Importing the libraries for the downstream processing'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Importing the dataset and reading age and salary to X, let's assume we are just looking for this only
for Simplicity lets take 2 variables in the X for plotting in 2 Dimension
'''
df = pd.read_csv('delivery.csv')
X = df.iloc[:, [2, 3]]
y = df.iloc[:, 4]

'''
Splitting the dataset into the Training set and Test set
'''
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
Feature Scaling needs to be performed as the values need to be normalized
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


'''
# Fitting Logistic Regression to the Training set
The OVR ( or the One over rest is leveraged for the calculation for evry class)
'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0,multi_class='multinomial')

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

'''
# Predicting the Test set results
'''
y_pred = classifier.predict(X_test)

'''
#**********************Evaluating/Validating the model on Ytest ad Ypred***********************************

# Making the Confusion Matrix
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
The score could be validated by adding the diagonal values
The score is 71 or 71% accuracy of the logistic model
'''

classifier.score(X_test, y_test)


'''
# Visualising the Test set results
'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','yellow','brown')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','yellow','brown'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Distance')
plt.ylabel('Parcel Amount')
plt.legend()
plt.show()


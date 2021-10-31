'''# K-Nearest Neighbors (K-NN) with Grid search for the optimization of the model
Lets try the elbow method for the optimal value of K
'''
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
#**Evaluating the model using the Grid Search for best value of k and choice of distance on train data***
# We are training the data for knn=3, lets see if the elbow method 
is able to identify the best knn for train data
'''
from sklearn.neighbors import KNeighborsClassifier


error_rate = []
for i in range(1,21):
 classifier = KNeighborsClassifier(n_neighbors=i)
 classifier.fit(X_train,y_train)
 pred_i = classifier.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))
 
 
 '''
 Lets plot the value of k and the error rate for optimal k selection
 
 What is the optimal vale of K is it 3? or 5 ?
 '''
 
plt.figure(figsize=(10,6))
plt.plot(range(1,21),error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


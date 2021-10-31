'''K-Means Clustering for finding the customer cluster for understaning the 
spending patterns and targeting the right customers. This technique is used in CRM for customer segmentation
or bucketing the right customers
'''
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

'''Importing the dataset and creating the X array with the 2 variables( annual income and customer score), 
Now we do not have Y as this is unsupervised  learning example
'''
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


'''Using the elbow method to find the optimal number of clusters or the optimal k value
Here we are going with 10 clusters as an arbitary choice and see what is the optimal size of the clusters
What is the optimal size of clusters, it is 5, after k=5 there is no significant change in WCSS
The curve also takes the elbow or curve at k=5
thus all mall customers should be divided into 5 categories
'''

'''Initializing the null list '''
'''Calling the class here and creating the object of the respective class
    We are fitting the model with k=1 to 10, init is k-means++ for avaoiding random initialisation trap'''
'''inertia is the other name of wcss in scikit learn and helps to calculate the distance
we are appending the distance to wcsss for all values of k (1-10)
'''

from sklearn.cluster import KMeans
wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

'''Plotting the value of K on X axis and the WCSS on y axis for finding the optimal k'''
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the mall dataset with 5 clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
# The fit predict method now seggregates the data in 5 clusters based on the KMeans algo
# The one dimensional array is also called as the vector
y_kmeans = kmeans.fit_predict(X)

# Using the kmeans for the new data
new = pd.read_csv('Mall_Customers new.csv')
Xnew = new.iloc[:, [3, 4]].values
y_kmeansnew = kmeans.fit_predict(Xnew)

# Visualising the clusters for the new data that we have trained
plt.scatter(Xnew[y_kmeansnew == 0, 0], Xnew[y_kmeansnew == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(Xnew[y_kmeansnew == 1, 0], Xnew[y_kmeansnew == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(Xnew[y_kmeansnew == 2, 0], Xnew[y_kmeansnew == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(Xnew[y_kmeansnew == 3, 0], Xnew[y_kmeansnew == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(Xnew[y_kmeansnew == 4, 0], Xnew[y_kmeansnew == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# This is for the centroid with size=150 to see it as a big one
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


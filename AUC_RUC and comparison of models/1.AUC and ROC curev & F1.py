'''
# Support Vector Machine (SVM) for the classification
# Importing the libraries
'''
import matplotlib.pyplot as plt
import pandas as pd

'''
# Importing the dataset from the csv
'''
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

'''
# Splitting the dataset into the Training set and Test set
'''
from sklearn.model_selection  import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
# Feature Scaling for reducing the impact of one variable on another
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
# Fitting SVM to the Training set with the rbf kernel, make sure you keep Probability=True
'''
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0,probability=True)
classifier.fit(X_train, y_train)

'''
# Importing the metrics from the scikitlearn library
Calculate the fpr and tpr for all thresholds of the classification, the probability index would be a  (N,2) matrix and 
this would help in classifying under 0 and 1, the sum of both the probabilities would be 1 or 100%'''

import sklearn.metrics as metrics
probs = classifier.predict_proba(X_test)

'''Just keeping one set , the other set could be easily calculated for the data
As they both are complimentary: The sum should be 1 or 100%

This now predicts the probability of 0 or "No"

Look at the first row the probability is 93% for column 0, thus the probability of column 1 is 7% approx

This can also be leveraged for a scorecard creation and other score problems
'''
preds = probs[:,1]

'''Calculating the fpr and tpr for all the possible value of preds or probability
Thresholds are decided on the basis of data max T =1+max

The preds can be altered to get the fpr and tpr at various levels
'''

fpr,tpr,threshold = metrics.roc_curve(y_test, preds)

y_pred = classifier.predict(X_test)

'''F1 score gives the trade off in Precision and Recal '''

f1=metrics.f1_score(y_test, y_pred)

''' Observe the max threshold.. it takes it as max+1.. both are same''''

m=max(preds)+1
max_T=max(threshold)


'''Calculating the AUC metric from fpr and tpr and it comes out to be 96%
The ROC is the blue points joines by the line.
'''
roc_auc = metrics.auc(fpr, tpr)

'''
# Method I: Plotting the ROC plot for all the probabilities and calculating the AUC as well
The Blue is the ROC for the classifier and the AUC is 96%, this AUC is very decent
'''

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')  #for red line with dots, just giveing red would give a filled line g-- would give a green line etc***
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('TPR: True Positive Rate')
plt.xlabel('FPR: False Positive Rate')
plt.show()



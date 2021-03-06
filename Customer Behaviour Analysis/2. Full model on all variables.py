'''Logistic Regression for predicting the event, this is best used as a Binary classifier
Importing the libraries for the downstream processing'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''Importing the dataset and reading age and salary to X, let's assume we are just looking for this only
for cimplicity lets take 2 variables in the X for plotting in 2 Dimension
Age and Salary as X
'''
df = pd.read_csv('Social_Network_Ads.csv')
X1 = df.iloc[:, [2, 3]].values
X2=df[['Gender']]

''' The Dummy variables should be created for the categorical variable'''
from sklearn.preprocessing import LabelEncoder

X2=X2.apply(LabelEncoder().fit_transform)

X=np.concatenate((X1,X2),axis = 1)
del X1,X2
y = df.iloc[:, 4].values

'''Splitting the dataset into the Training set and Test set'''
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''Feature Scaling needs to be performed as the values need to be normalized
why are we not scaling y?
Y is 0 and 1 thus no need for scaling
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
# Fitting Logistic Regression to the Training set
'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
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

'''Validating the Regression Model using the score and it is 91%
How can you validate the 91% score manually from the confusion matrix
This is the accuracy score of the model, it is able to classify data by accuracy of 89%

0     1 (Pred)

0  65    3
1  6    26

(Actual)
'''
'''
# The score has improved by adding one more variable in Logistic Regression
'''
classifier.score(X_test, y_test)

'''
# Calculating the TP,TN,FP and FN ,,, these are used for calculating specificity and sensitivity
'''
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

'''The Confusion matrix in the heatmap form for all combinations'''
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="viridis" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

'''
#Classification Accuracy: Overall, how often is the classifier correct? Does it match as above
'''
print((TP + TN) / float(TP + TN + FP + FN))
'''
#Classification Error: Overall, how often is the classifier incorrect? Also known as "Misclassification Rate"
# Why it is 11%
'''
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)

'''
#*******Sensitivity also known as Recall how good it is identifying the Positives**********
'''
sensitivity = TP / float(FN + TP)
print(sensitivity)
'''
#*******Specificity how good the model is identifying the Negatives**********
'''
specificity = TN / (TN + FP)
print(specificity)
'''
#********we can not plot as we have now 3 variables, we would need just 2 for plotting in 2D********
'''
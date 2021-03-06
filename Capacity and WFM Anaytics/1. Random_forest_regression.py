'''
# Random Forest Regression with Enseble Bagging
# Importing the libraries
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''# Importing the dataset'''
df = pd.read_csv('allocations.csv')
X1 = df.iloc[:, 0:3]
X2=  df.iloc[:, 3:4]
y = df.iloc[:, -1]

'''
# Encoding the Independent Variable Analyst Level, Complexity and data
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X1=X1.apply(LabelEncoder().fit_transform)

'''
#Adding the encoded data and the Height column and concatenating the X1 and X2 arrays
'''
X=pd.concat((X1,X2),axis = 1)

#Deleting the intermediate frames, these would no longer be used in the subsequent code
del X1,X2

'''
# Fitting Random Forest Regression to the dataset
# Creating an object, Fit and then Predict, Remember these three steps for all models :)
# N_estimators means how many Trees and we would take the average of all Trees, Trees in your Forest
# Default values is 10, Random_state is again the seed for the random number generator and is set to 0
'''
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0,oob_score=True)
regressor.fit(X, y)




'''
A random forest regressor. Let's explore the 16 hyperparameters, the most important for RF(Random Forest)

A random forest is a meta estimator that fits a number of classifying/Regression decision trees 
on various sub-samples of the dataset and uses averaging to improve
 the predictive accuracy and control over-fitting. 
The sub-sample size is always the same as the original input 
sample size but the samples are drawn with replacement 
if bootstrap=True (default value in teh RF ensemble model).

****************************
1. @@@@@
Parameters
n_estimators : integer, optional (default=10).. changing to 100 in new version
The number of trees in the forest.

Changed in version 0.20: The default value of n_estimators
will change from 10 in version 0.20 to 100 in version 0.22.
****************************
2. @@@@@
criterion : string, optional (default="mse")
The function to measure the quality of a split. Supported criteria are "mse" for the mean squared error,
 which is equal to variance reduction as feature selection criterion, and "mae" for the mean absolute error.

****************************
3. @@@@@
max_depth : integer or None, optional (default=None)
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all 
leaves contain less than min_samples_split samples.

max_depth=1 would only split the tree to 1 and then it would stop

****************************
4. @@@@@
min_samples_split : int, float, optional (default=2)
The minimum number of samples required to split an internal node:
If int, then consider min_samples_split as the minimum number.
If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number
of samples for each split.
Changed in version 0.18: Added float values for fractions.
****************************
5. @@@@@
min_samples_leaf : int, float, optional (default=1)
The minimum number of samples required to be at a leaf node. A split point at any depth will only be
considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. 
This may have the effect of smoothing the model, especially in regression.
If int, then consider min_samples_leaf as the minimum number.
If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number
 of samples for each node.
Changed in version 0.18: Added float values for fractions.

****************************
6.
min_weight_fraction_leaf : float, optional (default=0.)
The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. 
Samples have equal weight when sample_weight is not provided.

****************************
7. @@@@@
max_features : int, float, string or None, optional (default="auto")
The number of features to consider when looking for the best split:
If int, then consider max_features features at each split.
If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
If "auto", then max_features=n_features.
If "sqrt", then max_features=sqrt(n_features).
If "log2", then max_features=log2(n_features).
If None, then max_features=n_features.
Note: the search for a split does not stop until at least one valid partition of the node samples is found, 
even if it requires to effectively inspect more than max_features features.
****************************
8.
max_leaf_nodes : int or None, optional (default=None)
Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. 
If None then unlimited number of leaf nodes.

****************************
9.
min_impurity_decrease : float, optional (default=0.)
A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

The weighted impurity decrease equation is the following:

N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
where N is the total number of samples, N_t is the number of samples at the current node,
 N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.

N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.

New in version 0.19.
****************************
10.
min_impurity_split : float, (default=1e-7)
Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, 
otherwise it is a leaf.

Deprecated since version 0.19: min_impurity_split has been deprecated in favor of min_impurity_decrease in 0.19.
 The default value of min_impurity_split will change from 1e-7 to 0 in 0.23 and it will be removed in 0.25. Use min_impurity_decrease instead.
****************************
11. @@@@@
bootstrap : boolean, optional (default=True)
Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
If the whole dataset is used the trees would be same 

****************************
12. @@@@@
oob_score : bool, optional (default=False)
whether to use out-of-bag samples to estimate the R^2 on unseen data.

****************************
13.
n_jobs : int or None, optional (default=None)
The number of jobs to run in parallel for both fit and predict.
 None` means 1 unless in a joblib.parallel_backend context.
 -1 means using all processors. See Glossary for more details.
 
 Basically for the processors to be used for computation
 
 ****************************
14. @@@@@
random_state : int, RandomState instance or None, optional (default=None)
If int, random_state is the seed used by the random number generator; 
If RandomState instance, random_state is the random number generator; 
If None, the random number generator is the RandomState instance used by np.random.
****************************
15.
verbose : int, optional (default=0)
Controls the verbosity when fitting and predicting.
****************************
16.
warm_start : bool, optional (default=False)
When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, 
otherwise, just fit a whole new forest. See the Glossary.

'''


'''Always Validate the RF model usingthe OOB score, which is Out Of Bag score
This is the score that tells how well the model fits on OOB observations
'''
print(regressor.oob_score_)



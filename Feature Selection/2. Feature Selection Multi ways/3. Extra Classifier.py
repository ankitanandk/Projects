'''
Lets try to get the different ways of Feature Selection in 1 shot
First Lets understand the data, Now we have 21 Variables and 2000 rows in Train and 1000 rows in test data

*************21 Variables*************************************

Y:
price_range: This is the target variable with a 4 values of

 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

The Y is Categorical/ordinal in nature.


X:
    
Description of variables in the above file
battery_power: Total energy a battery can store in one time measured in mAh
blue: Has Bluetooth or not
clock_speed: the speed at which microprocessor executes instructions
dual_sim: Has dual sim support or not
fc: Front Camera megapixels
four_g: Has 4G or not
int_memory: Internal Memory in Gigabytes
m_dep: Mobile Depth in cm
mobile_wt: Weight of mobile phone
n_cores: Number of cores of the processor
pc: Primary Camera megapixels
px_height
Pixel Resolution Height
px_width: Pixel Resolution Width
ram: Random Access Memory in MegaBytes
sc_h: Screen Height of mobile in cm
sc_w: Screen Width of mobile in cm
talk_time: the longest time that a single battery charge will last when you are
three_g: Has 3G or not
touch_screen: Has touch screen or not
wifi: Has wifi or not

'''
 

'''
 *************
3. Feature Importance :
You can get the feature importance of each feature of your dataset by using the feature
 importance property of the model.
Feature importance gives you a score for each feature of your data, the higher
 the score more important or relevant is the
 feature towards your output variable.
 
Feature importance is an inbuilt class that comes with Tree Based Classifiers, 
we will be using Extra Tree Classifier for
extracting the top 10 features for the dataset.

Please learn that the This class implements a meta estimator that fits a number of 
randomized decision trees (a.k.a. extra-trees) on various sub-samples
 of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

 ************************************* 
'''

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train_mobile.csv")
''' #independent columns the features or the X's'''
X = data.iloc[:,0:20]
'''#target column i.e price range'''
y = data.iloc[:,-1]    
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
'''#use inbuilt class feature_importances of tree based classifiers'''
print(model.feature_importances_) 
'''#plot graph of feature importances for better visualization'''
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()















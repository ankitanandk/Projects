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
 *************1 Chi Square for the Feature Selection************************************* 
'''
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv("train_mobile.csv")
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

'''Apply SelectKBest class to extract top 10 best features
the scoring Algo that is being used is Chi Square and we are interested in the top 10 features
These are the top 10 features that define the Y.

1. the k=10 basically means that you are interested in top 10 features thet explain or dependent on Y :)
'''
'''Creating the object from t he SelectKBest class'''
bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

'''The column level scores are listed using the chi sq algorithm, gives column index and chi square score
The chisquare score varies from 0 to Infinity 
'''
dfscores = pd.DataFrame(fit.scores_)

'''Gives the column names and their index'''
dfcolumns = pd.DataFrame(X.columns)

'''Concat two dataframes for better visualization and clarity, both are concatenated vertically '''
featureScores = pd.concat([dfcolumns,dfscores],axis=1)

'''renaming the dataframe columns for clarity, calling them as vars and score
Once these columns have been achieved, we can now try and make sense of the y using these variables :)
'''
featureScores.columns = ['Var_Name','Score']  
'''The nlargest gives the top 10 highest rows from a data frame'''
top_10=featureScores.nlargest(10,'Score')

'''Printing the top 10 scores for clarity'''
print(top_10)


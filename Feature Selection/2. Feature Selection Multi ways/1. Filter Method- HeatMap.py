'''
Lets try to get the different ways of Feature Selection in 1 shot
First Lets understand the data, Now we have 21 Variables and 2000 
rows in Train and 1000 rows in test data

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
 
import pandas as pd
data = pd.read_csv("train_mobile.csv")
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

'''
 *************
1.Correlation Matrix with Heatmap: Pearson Correlation
Correlation states how the features are related to each other or the target variable.
Correlation can be positive 
(increase in one value of feature increases the value of the target variable) or negative 
(increase in one value of feature decreases the value of the target variable)
Heatmap makes it easy to identify which features are most related to the target variable, 
we will plot heatmap of correlated features using the seaborn library.


This is only valid for Numeric X variables

 ************************************* 
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
'''#get correlations of each features in dataset'''
cor = data.corr()
top_corr_features = cor.index
plt.figure(figsize=(50,50))

'''#plot heat map with the seabor Module'''
sns.heatmap(cor,annot=True,cmap="RdYlGn")
plt.show()


'''#Correlation with output variable of each X'''
cor_target = abs(cor["price_range"])

'''sorting the series '''
cor_target=cor_target.sort_values(ascending=False)

'''Selecting highly correlated features that have a correlation value above 0.5
You can even check the last row of correlation matrix for the best correlated features


Looks like just RAM is the only feature with corr value above 0.5
'''
relevant_features_positive = cor_target[cor_target>0.5]
relevant_features_negative = cor_target[cor_target<-0.5]
print("positive features",relevant_features_positive)
print("negative features",relevant_features_negative)












'''
The Below code would give you the HIGH Level idea of features that are interacting with the "Y" and explain it.

The seaborn module give a nice heatmap with scale and again a clour grid
Even if you run for the frame, it would consider numeric variables
The heatmap method takes the corr values of df.corr matrix as input for the plot

'''
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
df=pd.read_csv('admit.csv')

'''#Using Pearson Correlation coefficient'''
plt.figure(figsize=(12,10))
'''The corr gives the exact correlation between two variables: Always Remember the Corr value lies between -1 and 1
The Below code would yield a matrix :)
 '''
cor = df.corr()

'''#For Blue colour '''
sb.heatmap(cor, annot=True, cmap=plt.cm.Blues)

'''#For Red colour'''
sb.heatmap(cor, annot=True, cmap=plt.cm.Reds)

'''Correlation with output variable Weight, 
lets try and see the feature/variable that explains the weight to maximum extent
This tells that weight is explained maximum by height for the given data :)
'''
'''Taking absolute value of Weight correlation
The Explanation could be Direct linear or Inverse linear
'''
cor_target = abs(cor["Weight"])

'''Selecting highly correlated features with correlation above 50%
You can choose any value: If you go higher, you are actually very strict in your selection
'''

relevant_features = cor_target[cor_target>0.5]
relevant_features

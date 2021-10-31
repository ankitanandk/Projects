'''
This would give the visualization for all the variable combinations that meet the cutoff

The seaborn module give a nice heatmap with scale and again a clour grid
Even if you run for the frame, it would consider numeric variables
The heatmap method takes the corr values of df.corr as input
'''
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
df=pd.read_csv('50_Startups.csv')

'''Using Pearson Correlation coefficient for all the combinations'''
plt.figure(figsize=(8,7))
'''The corr gives the corr values between -1 and 1: These value give an indication of the direction and strength of association
Ignore the Diagonal and rest of the combinations would give a high level idea of x that is associated with Y.
'''
cor = df.corr()
'''For Red colour and just looking at corr values above 0.8 (Positive correlation amongst all the variables)
Now just focus on the X's that are 80% or more correlated to Y and these are your potential predictors
'''
x=sb.heatmap(cor, annot=True, cmap=plt.cm.Reds,vmin=0.8)

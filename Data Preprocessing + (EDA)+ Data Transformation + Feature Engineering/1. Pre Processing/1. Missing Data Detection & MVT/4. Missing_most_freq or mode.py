'''
#1.Data processing, below are the libraries we that we would need for all ML sessions
'''
import pandas as pd
import numpy as np
'''
# Importing the dataset, remember to set a working directory in Spyder
#Separating data in X(country, age and sal) and Y(Purchased)
#For printing you can use the console as well
'''

df = pd.read_csv('Data.csv')
x = df.iloc[:, :1].values


'''
# for each column, get value counts in decreasing order and take the index (value) of most common class

char var

'''
newx = df.apply(lambda x: x.fillna(x.value_counts().index[0]))


'''
Hint on using the value count function available for a series or a column of a frame :)

'''
df['Country'].value_counts()



'''
#1.Data processing, below are the libraries we that we would need for all ML sessions

LOCF: Last observation carried forward :)
'''
import pandas as pd
'''
# Importing the dataset, remember to set a working directory in Spyder
#Separating data in X(country, age and sal) and Y(Purchased)
#For printing you can use the console as well
'''

df = pd.read_csv('Data.csv')
X = df.iloc[:, 1:-1]

'''
This would lead to data loss as it drops the rows where any variable value is misisng :)

'''

x_new=X.dropna()
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

'''
# replace missing values with the column mean

'''
df_mean_imputed = df.fillna(df.mean())
df_median_imputed = df.fillna(df.median())


'''
Frame after the imputation

'''
print(df_mean_imputed)

'''
Frame after the imputation

'''

print(df_median_imputed)
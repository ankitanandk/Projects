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
#for forward-fill

'''
X_fwd=X.fillna(method='ffill')


'''
For Back-fill or forward-fill to propagate next or previous values respectively:
#for back fill 
 
'''
X_back=X.fillna(method='bfill')


'''
#one can also specify an axis to propagate (1 is for rows and 0 is for columns)

'''

x_new_1=X.fillna(method='bfill', axis=1)


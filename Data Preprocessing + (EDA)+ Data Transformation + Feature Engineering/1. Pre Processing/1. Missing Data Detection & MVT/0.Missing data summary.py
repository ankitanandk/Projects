'''#1.Data processing, below are the libraries we that we would need for all ML sessions
'''
import pandas as pd

'''
# Importing the dataset, remember to set a working directory in Spyder
#Separating data in X(country, age and sal) and Y(Purchased)
#For printing you can use the console as well
'''
df = pd.read_csv('Data.csv')

'''
df.info() gives the summary that helps in summarizing the data :)

'''

df.info()


'''
the isnul().sum() function could be applied to get the details of the missing values
'''

df.isnull().sum()

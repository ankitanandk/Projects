'''The below code shows how to show the scatter of whole data
Note only the numeric variables contibute to the matrix 
The scatter_matrix function is in plotting module of pandas
'''
import pandas as pd
df=pd.read_csv('admit.csv')
pd.plotting.scatter_matrix(df,figsize=(10,10))



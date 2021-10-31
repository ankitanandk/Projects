# The below code shows how to plot between all the numeric variables, just like a matrix or a grid
import pandas as pd
import seaborn as sb
df=pd.read_csv('admit.csv')
#Reading all the numeric columns into the dataframe
df=df._get_numeric_data()
'''Lets look at the scatter plot using the seaborn module, this gives a pair between all variables, scroll till bottom :)
This is a cross plot, focus on the diagonal,its in a matrix form
only numeric variables are being used to plot, we would see later 
what would happen if we push the entire dataframe'''
sb.pairplot(df)
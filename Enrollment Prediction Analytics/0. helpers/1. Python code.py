#Importing various modules, Try this code on Jupyter for better visuals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter


%matplotlib inline
rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')

#Reading the enrollment csv from the path, all variables should be numeric and continous
df=pd.read_csv('enroll.csv')
df.head()

#Lets look at the scatter plot using the seaborn module, this gives a pair between all variables, scroll till bottom :)
# This is a cross plot, focus on the diagonal,its in a matrix form
sb.pairplot(df)

#Lets look at the correlation, 
#look at the value unemployment vs highschoolgrade 0.177, there is almost no correlation, thats a good news
# we dont want  
print (df.corr())

#Lets pull the x and y clumns from frame, the unem and hgrad are X and the inc is our Y
df_enroll=df.ix[:,(2,3)].values
df_target=df.ix[:,1].values

enroll_data_names=['unem','hgrad']

#Scaling the X variables, we already know why we scale, just to avoid the eucledian distance issue
X,y=scale(df_enroll), df_target

#Check for missing values, if it does not return anything that means there are no missing values
missing_val=X==np.nan
X[missing_val==True]

# Creating the object
reg=LinearRegression(normalize=True)
reg.fit(X,y)

#Lets print the r square score
print(reg.score(X,y))
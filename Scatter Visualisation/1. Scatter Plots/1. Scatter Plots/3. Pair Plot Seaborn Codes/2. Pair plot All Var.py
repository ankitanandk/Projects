# The below code shows how to plot between all the numeric variables, just like a matrix or a grid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df=pd.read_csv('admit.csv')

'''What would happen if I include a all variables(Mix of Continous and categorical), 
it would basically ignore them and plot only numerical

Lets look at the scatter plot using the seaborn module, this gives a pair between
 all variables, scroll till bottom :)
This is a cross plot, focus on the diagonal, it is between the same variable on x and y axis,
its in a matrix form'''
sb.pairplot(df)

# ANother way of writing the code in a simplistic way :) and the relation
import pandas as pd
import matplotlib.pyplot as plt
#Reading the csv to a dataframe
df=pd.read_csv(r'C:\Users\ankit\Downloads\ML Python\Classes\Class 28\Notes\1. Scatter Plots\1. Scatter Plots\2. Scatter Plot Codes\salary.csv')
# Plottting scatter using the scatter function from matplotlib
plt.scatter(df.YearsExperience,df.Salary, c="green")
plt.title('Scatter plot of years and Salary')
plt.xlabel('Exp in years')
plt.ylabel('Sal in $')
#Prints the plot instead of sending to the output
plt.show()
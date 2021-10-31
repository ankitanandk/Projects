#The below code shows how to plot between x and y and create a scatter plot for admit data
import pandas as pd
import matplotlib.pyplot as plt
#Reading the csv to a dataframe
df=pd.read_csv(r'C:\Users\ankit\Downloads\ML Python\Classes\Class 28\Notes\1. Scatter Plots\1. Scatter Plots\2. Scatter Plot Codes\admit.csv')

# Plottting scatter using the scatter function from matplotlib, the Age is now x and Height is Y
plt.scatter(df.Age, df.Height,c="Red")
plt.title('Scatter plot age and height')
plt.xlabel('Age in years')
plt.ylabel('Height in cms')
 # This helps to show the current graph, if we dont use it, graph still comes with the last command in output as well.
plt.show() 
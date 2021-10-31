# Regression plot using Seaborn library, regplot helps in creating a beautiful graph, with different options
# By default it creates the Regression line, as the defauly value is True for the Regression line :)
import seaborn as sb
import pandas as pd
df =pd.read_csv('salary.csv')

# use the function regplot to make a scatterplot without regline, just the data points
sb.regplot(x=df.YearsExperience,y=df.Salary,color="Orange",marker="x",fit_reg=False)


# use the function regplot to make a scatterplot
sb.regplot(x=df.YearsExperience,y=df.Salary,color="Orange",marker="x")


 
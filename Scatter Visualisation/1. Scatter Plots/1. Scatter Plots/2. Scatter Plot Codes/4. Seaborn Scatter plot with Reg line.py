# Regression plot using Seaborn library, regplot helps in creating a beautiful graph
# By default it creates the Regression line, as the defauly value is True for the Regression line :)
import seaborn as sb
import pandas as pd
df =pd.read_csv('admit.csv')

# use the function regplot to make a scatterplot
sb.regplot(x=df.Age, y=df.Height)
# Regression plot using Seaborn library, regplot helps in creating a beautiful graph in 2 dimenson
# The Regression line that comes by default can be avoided by using fit_reg=False
import seaborn as sb
import pandas as pd
df =pd.read_csv('admit.csv')

# use the function regplot to make a scatterplot with x and y series, fit_reg=False would not give Identity line
sb.regplot(x=df.Age, y=df.Height, fit_reg=False)

import numpy as np

a=np.array([[[1,2],[3,4]]])
print(a.ndim,a.shape)
print(a)
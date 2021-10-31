#importing pandas,numpy and seaborn libraries
import pandas as pd
import numpy as np
import seaborn as sb

#importing the data from saved excel file for price after x years and selling price
#Sheetname for reading the sheet precisely
df=pd.read_excel('price.xlsx',sheetname='depreciation')
print(df.shape)
df.head()

#collecting x and y data from the data frame to an one dimension numpy array for calculation ease
X=df['Year'].values; 
Y=df['Sell_Percentage'].values

#Calculating the mean of x and y
mean_x=np.mean(X); mean_y=np.mean(Y)

#Total number of values captured in  a array, basically the number of rows, this would help in loop iterations
n=len(X) 

#calculating the linear regression coefficients, the slope(b) and the y Intercept(a)
numer=0
denom=0
for i in range(n):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y) # Covariance of X and Y, this is the numerator
    denom+=(X[i]-mean_x)**2 # Variance of X, this is the denominator
b=numer/denom #equation of calculating slope
a=mean_y-(b*mean_x) #equation of y intercept
    
#print the coefficients of the best fit line, the model say at 0 year price=77.5%
#the rateof change per year is almost -6.5%
print(a,b)

#plotting regression line and the actual observed point using the seaborn module
sb.regplot(x=df["Year"], y=df["Sell_Percentage"])

#Calculate the r square, for the goodness of fit, that helps in undersating how much the model explains the data points
#The R square comes out to be .9768, this means 97.68% of the variation is explained by the model :)
sse=0 # Sum of squared errors
sst=0 # Sumof squared Total
for i in range(n):
    y_pred=a+b*X[i]
    sst+=(Y[i]-mean_y)**2
    sse+=(Y[i]-y_pred)**2
    r2=1-(sse/sst)
    
print(r2)
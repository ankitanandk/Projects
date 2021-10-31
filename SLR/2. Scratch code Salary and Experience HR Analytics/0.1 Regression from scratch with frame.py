'''importing pandas,numpy and seaborn modules for further use in reading data and plotting

If a module does not gets used you would see a warning message :)
'''
import pandas as pd
import numpy as np
import seaborn as sb

'''importing the data from saved csv file,
 if you use the path do use "r" for reading the path as raw string'''
df=pd.read_csv('Salary.csv')
#look at the shape in console
print(df.shape)

'''collecting X and Y data from the data frame to series.. 
Series is essentially a column of frame..
'''
X=df['YearsExperience']
Y=df['Salary']

'''Look at the console for the type of X and its dimension, it is a pandas series and 1 dimension :)'''
print(type(X),X.ndim)

'''Arithmetic mean of x and y variables, using the mean function from numpy'''
mean_x=np.mean(X)
mean_y=np.mean(Y)

'''To print the captured mean of x and y for clarity, look at console  or at variable explorer:)'''
print(round(mean_x,2),round(mean_y,2)) 

'''Total number of values captured in  a series, basically the number of rows, this would help in loop iterations
With the range function 
'''
n=len(X) 

'''
#calculating the linear regression coefficients,the slope(b) and the y Intercept(a)
#Lets first try and calculate the numerator and denominator for slope
'''
numer=0;
denom=0
a=range(n)
for i in range(n):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y) # Covariance of X and Y, this is the numerator of the slope calculation
    denom+=(X[i]-mean_x)**2 # Variance of X, this is the denominator for slope calculation'

b=numer/denom #equation of calculating slope of best fit line using OLS (Ordinary Least Squares)
a=mean_y-(b*mean_x) #equation of y intercept of best fit line using OLS (Ordinary Least Squares)
    
'''#print the coefficients of the best fit line, are they same as Excel calculations?'''
print(a,b)

'''#Equation for the salary forecast of 13 years of Experience'''
y_sal13=a+b*13; print(y_sal13)

'''#plotting regression line and the actual observed point using the seaborn module'''
sb.regplot(x=df["YearsExperience"], y=df["Salary"])

'''
#Calculate the r square, for the goodness of fit, that helps in undersating how much the model explains the data points
#The R sqaure comes out to be .9569, this means 95.69% of the variation is explained by the model :)

'''
sse=0 # Sum of squared errors(y-yhat)... the unexplained part
sst=0 # Sumof squared Total(y-ybar).. The totel error
ssr=0 #Sum of squared regression... The explained part 
rmse=0
for i in range(n):
    yhat=a+b*X[i]
    sst+=(Y[i]-mean_y)**2
    sse+=(Y[i]-yhat)**2
    rmse+=(((Y[i]-yhat)**2)/n-2)
    ssr+=(yhat-mean_y)**2
print(rmse**0.5) #Taking the underroot of the value, this now tells the interval estimates, similar to sigma
r2=1-(sse/sst)
r2_new=ssr/sst

'''The valye of r2 from both the calculations is SAME>.. >)   .9569'''
print(r2,r2_new)

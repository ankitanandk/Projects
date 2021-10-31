#importing pandas,numpy and matplotlib libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the data from saved csv file, r is used with path to read the raw path and avaoid any special meaning
df=pd.read_csv('headbrain.csv')
print(df.shape)
df.head()

#collecting x and y data from the data frame to an one dimension numpy array, look atthe size in Variable explorer
X=df['Head Size(cm^3)'].values
Y=df['Brain Weight(grams)'].values

mean_x=np.mean(X)
mean_y=np.mean(Y)

#To print the captured mean of x and y for clarity, look at console :)
print(mean_x,mean_y) 

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
    
#print the coefficients of the best fit line
print(a,b)

#plotting values and Reg line, calculating max and min of X for the defining the scale
max_x=np.max(X)+100
min_x=np.min(X)-100
print(max_x,min_x)

#Calculating line values x and y, this means start from min and go till max and create 100 uniform spaced values
#The whole deal is we are trying to create many x values and for each x we are predicting the y using the coefficients
#all in all we are trying to create NEW DATA values for the Reg line, consider this as the NEW DATA
x=np.linspace(min_x,max_x,100)
y=a+b*x
plt.scatter(X,Y,c='Red',label="Scater Plot actual data 237 X,Y data points")

#plotting regressionline and the actual observed point
plt.plot(x,y,color='green',label="Regression Line")
plt.xlabel('head size')
plt.ylabel('Brain weight')
plt.legend()
plt.show()

#Calculate the r square, for the goodness of fit, that helps in undersating how much the model explains the data points
#The R sqaure comes out to be .639, this means 63.9% of the variation is explained by the model :)
sse=0 # Sum of squared errors
sst=0 # Sumof squared Total
for i in range(n):
    y_pred=a+b*X[i]
    sst+=(Y[i]-mean_y)**2
    sse+=(Y[i]-y_pred)**2
    r2=1-(sse/sst)
print(r2)

'''Clearly the Inflexion point is at Degree=5 as the elbow is formed at this point.

The gainin R2 is non significant or reaches a plateau after 5

Thus the order of the polynomial is 5: The index starts at 2

y=a+b1x+b1x2+b3x3+b4x4


Thus the roots or factors would be 5 and we have fit the data well as compared to a SLR
'''

import matplotlib.pyplot as plt

'''Link for details as below

https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
'''

li=[]
for i in range(2,10):
    poly_reg = PolynomialFeatures(degree = i)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression() 
    lin_reg_2.fit(X_poly, y)
    r2_score_poly=lin_reg_2.score(X_poly,y)  
    li.append(r2_score_poly)
    
plt.plot(np.arange(2,10),li, marker='o',color = 'blue', linewidth=3, linestyle='dashed',markersize=12)


for i, j in enumerate(li,2):
    print(f"The degree is {i} and the r2 is {j}")
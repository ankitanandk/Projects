'''Clearly the Inflexion point is at Degree=3 as the elbow is formed at this point

Thus the order of the polynomial is 3

y=a+bx+bx2+bx3


Thus the roots or factors would be 3 and we have fit the data well as compared to a SLR
'''

import matplotlib.pyplot as plt

'''Link for details as below

https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
'''

li=[]
for i in range(2,6):
    poly_reg = PolynomialFeatures(degree = i)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression() 
    lin_reg_2.fit(X_poly, y)
    r2_score_poly=lin_reg_2.score(X_poly,y)  
    li.append(r2_score_poly)
    
plt.plot(np.arange(2,6),li, marker='o',color = 'blue', linewidth=3, linestyle='dashed',markersize=12)

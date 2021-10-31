import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("mobile_reg.csv")
''' #independent columns the features or the X's'''
X = data.iloc[:,0:20]
'''#target column i.e price range'''
y = data.iloc[:,-1]    

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
'''#use inbuilt class feature_importances of tree based classifiers'''
print(model.feature_importances_) 
'''#plot graph of feature importances for better visualization'''
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

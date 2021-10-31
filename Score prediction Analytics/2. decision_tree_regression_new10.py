
'''
###############################################################################################

# We have data of family income, distance from schoold and study hours for 10 new students
# Let's try and predict the scores of the students

We can use the trained regressor object for predicting scores

##############################################################################################

'''
df_new10 = pd.read_csv('student_scores_new10.csv')

X_new10= df_new10.iloc[:, 0:3].values

'''
# Using the trained regressor on the new data points, The Y or the predicted values are ready !!
'''
y_pred_new10=regressor.predict(X_new10)  
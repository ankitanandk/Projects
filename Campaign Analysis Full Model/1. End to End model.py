'''
*********************
STEP 0: Import all libraries for processing the data
*********************
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

'''
*********************
Data Import for the model

STEP 1: Import the data and have a clear understanding of the X(s) and Y
Know you data.... if the business problem is not clear the model making wont be easy


Data is related to direct marketing campaigns (phone calls) of a banking institution.
The classification goal is to predict whether the client will subscribe (1/0) to a term deposit (variable y). 

*********************
'''

'''Reading the bank.csv data which has 41188 observations and 16 features or columns
Make sure when you download, try to transpose and make the format as correct
'''
df=pd.read_csv('bank.csv',header=0)
df.shape

'''
*********************
STEP 1.1: Try and get the sense of the variable distribution..
How many are char.. how many are numeric..

The info method comes in very hand and gives a good summary..

Read the help for more details and understanding :0
*********************
'''

df.info()

'''
Input variables: 15 X variables and 41188 observations
***************
1. age (numeric)
2. job : type of job (categorical: “admin”, “blue-collar”, “entrepreneur”, “housemaid”, “management”, “retired”, “self-employed”, “services”, “student”, “technician”, “unemployed”, “unknown”)
3. marital : marital status (categorical: “divorced”, “married”, “single”, “unknown”)
4. education (categorical: “basic.4y”, “basic.6y”, “basic.9y”, “high.school”, “illiterate”, “professional.course”, “university.degree”, “unknown”)
5. default: has credit in default? (categorical: “no”, “yes”, “unknown”)
6. housing: has housing loan? (categorical: “no”, “yes”, “unknown”)
7. loan: has personal loan? (categorical: “no”, “yes”, “unknown”)
8. contact: contact communication type (categorical: “cellular”, “telephone”)
9. month: last contact month of year (categorical: “jan”, “feb”, “mar”, …, “nov”, “dec”)
10. day_of_week: last contact day of the week (categorical: “mon”, “tue”, “wed”, “thu”, “fri”)
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’). The duration is not known before a call is performed, also, after the end of the call, y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model
12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15. poutcome: outcome of the previous marketing campaign (categorical: “failure”, “nonexistent”, “success”)

'''

'''
Predict variable (desired target)
y — has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)
'''

'''
*********************
STEP 1.2: Try and identify the features that have a sigma/std of 0
that basically means a numeric variable with a static value and that may not be required

Drop the features as they may not be of any use in the final model.

Right Now the std value is not 0 for any column....

If the need be we can push in a list and drop them from the main frame

*********************
'''
d1=df.describe(include="number").loc['std']
d1_0=d1.to_frame(name='std').query("std==0")

'''
Dropping the columns if std=0
Right now it would not as none of the columns have std==0
'''
df.drop([i for i in d1_0.index], axis = 1) 


'''
*********************
STEP 1.3: Try and identify the features that are objects and 
unique value count is very high

These values would pose issues in grouping the data.

By inspection there are 3 features with more than 5 distinct values.

Job Education and Month need to be clubbed togethre to avoid multiple classes and dummy vars
*********************
'''

d2=df.describe(include="object").loc['unique']
d2_0=d2.to_frame(name='unique').query("unique>5")

df['education'].unique()

'''The where is used to replace the values just like if else
Thus the count is reduced from 8 to 6
Lets us groups the categories together for avoiding the dummy variable creation issue
Let us group the similar categories together
Clubbing all basic.4 6 and 9 to basic
'''

df['education']=np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.6y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.4y', 'Basic', df['education'])

'''The categories now have reduced since we clubbed 3 to 1 'Basic' '''
df['education'].unique()

'''
*********************
STEP 1.4: Try and identify the features that are either
primary key and they need to be removed from model.
*********************
'''

'''
*********************
STEP 1.5: Try and identify the features that are dates, 
we may not need them based on the requirement
*********************
'''


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

'''
*********************
STEP 2: Removing the missing observations, and Perform MVT
looks like no observation is missing and it works in our favour
*********************
'''
df_miss=df.isnull().sum()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

'''
*********************
STEP 3: Perform the outlier identification and treatment as explained in previous sections 
*********************
'''


'''
*********************
STEP 4: Perform the Feature selection using the various algorithms that we have discussed
*********************
'''


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

'''
*********************
STEP 5: Perform various viualizations for a better glimpse of data
*********************
'''

df.groupby('y').mean()

'''
Observations:
1.The average age of customers who bought the term deposit is 
higher than that of the customers who didn’t.

2. Surprisingly, campaigns (number of contacts or calls made during the current campaign)
 are lower for customers who bought the term deposit.
'''

'''
Job seems a good x
'''
pd.crosstab(df.job,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')


'''
marital seems not so good x
'''
table=pd.crosstab(df.marital,df.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')

'''
education seems a  good x
'''

table=pd.crosstab(df.education,df.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')

'''
day of week seems NOT a  good x
'''

pd.crosstab(df.day_of_week,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')

'''
month seems  a  good x
'''

pd.crosstab(df.month,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

del d1,d1_0,d2,d2_0,df_miss,table

'''
*********************
STEP 6: Dummy variable creation


Not needed for Decision Trees would be handy for logistic regression
*********************
'''


cat_vars=['job','marital','education','default','housing','loan',
          'contact','month','day_of_week','poutcome']
for var in cat_vars:
    
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    df1=df.join(cat_list)
    df=df1

    
df_vars=df.columns.values.tolist()
to_keep=[i for i in df_vars if i not in cat_vars]


data_final=df[to_keep]

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


'''
*********************
STEP 7: Final Model creation
Decision Tree Classification
*********************
'''

del cat_list,cat_vars,data_final,df1,df_vars,to_keep,var




X = df.iloc[:, [i for i in range(0,15)]]
y = df.iloc[:, [-1]]


''' Keeping the char for label encoding '''

char=X.select_dtypes(exclude=[np.number])

num=X._get_numeric_data()

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

char = char.apply(LabelEncoder().fit_transform)


''' Joining the char and num back for the complete: X '''


x=pd.concat([char,num],axis=1)

del X,char,num


   
'''# Splitting the dataset into the Training set and Test set'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


'''# Fitting Decision Tree Classification to the Training set'''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)


'''# Predicting the Test set results'''
y_pred = classifier.predict(X_test)

'''# Making the Confusion Matrix: what is the accuracy and error score for the confusion Matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''Lets calculate the accuracy score of the confusion matrix using the function'''
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


'''***************************************************************************************************************'''

'''Another way to visualize the CM with labels of TP FP FN and TN'''

from sklearn.metrics import confusion_matrix
pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted 0 (TN)' , 'Predicted 1 (FP)'],
    index=['Actual 0 (FN)', 'Actual 1 (TP)']
)

'''***************************************************************************************************************'''



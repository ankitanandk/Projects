'''
# Apriori : Association Rule Algorithm: This works on Support, Confidence and the Lift

However, scikit-learn does not support this algorithm. Fortunately, the very useful MLxtend library by
 Sebastian Raschka has a a an implementation of the Apriori algorithm for 
extracting frequent item sets for further analysis.

***************************
This is a weekly purchase data of a French grocery store with 7500 transactions
***************************
'''

''' Importing the library for reading the data and creating the rules'''
import pandas as pd

'''Data Preprocessing and reading the file into a data frame
Header= None is used so that the items purchased are not treated as variables
This is a weekly purchase data of a French grocery store with 7500 transactions
Each row is the basket of the customer that came for the shopping
'''
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


'''
Let's have a small dummy code and understand what it does?
Take 2 mins and understand each and every aspect of it, this would help in understanding the Apriori
The Apriori expects the input as list of list in a string format
'''

import pandas as pd
df = pd.DataFrame({'id': [1, 3, 5],'marks': [20, 40, 60]})

l=[]
for i in range(0, 3):
    l.append([str((df.values[i,j])) for j in range(0, 2)])    
print(l)


'''
The Apriori algorithm expects the following data before starting to make the rules

1. The data should be in a list and individual transactions should be also in list, so list of list
2. The values should be in strings, else it treats them as varibales or features
'''
'''Declaring an empty list for adding all the elements and iterarting over all rows and columns
And then deleting the i variable
'''

transactions = [] 
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
del i

'''Training Apriori on the dataset, we are imprting the apriori class from the module apyori
This is the weekly sales data for the grocery store

1. Min_support=.003: That basically means a product purchased 3 times a day or 21 times a week
Thus min_support=3*7/7500 ~ .003

If you are increasing the minimum suport, you are looking for prducts that are sold more frequently

2. Min_confidence: That is most likely to buy 20% of the times, its a thumb rule

3. min_lift: 3 is a good lift criteria, for good rules keep it high, again a thumb rule

4. min_length: How many min items you wish to see in the basket, we usually go for 2. Like how Item2 support Item1

'''
from apyori import apriori


rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

'''Visualising the results, as the object is a generator object, converting to a list
A total of top 154 rules have been created from the 7500 week transactions

The Rules are ordered and basically all 3 parameters combined give the rule order, just not 1 factor
'''
'''The class is a generator'''
print(type(rules))

results = list(rules)

''' Let's see the top 5 rules and what 2 items move together'''
for i in range(0,20):
    print(f"***{i}***{results[i]}")
    print("######################")
    
    
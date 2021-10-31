    '''
    Natural Language Processing: Always try and import a file with a unique delimeter. 
    CSV would not be a good choice as the comments or text could have comma in them.
    
    The ideal choice would be a tsv or the tab separated values or a delimeter that should not be found in the text comments.
    '''
    # Importing the library pandas for reading the csv file
    import pandas as pd
    
    '''
    STEP 1 *********
    
    Importing a tsv and taking care of the quotes
    
    Importing the dataset and quoting=3: Ignores the double quotes in the file.
    quoting=3 ignores the double quote in the text and also if text contains the
    delimeter it wraps in quotes
    This helps to avoid any potential issue we may face later in processing
    '''
    
    
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


'''    
STEP 2 *********: 

Cleaning the texts is the second step in NLP: 
        
1) re : The Regular Expression library. For searching the text patterns
It is used to remove the alphabets or numbers that may not be releveant to the analysis, 
rather than specifying what to remove, we specify what to keep ^a-zA-Z, 
it means just keep alphabets in all cases.. upper and lowercase

2) The space helps to act as delimeter after the removal of character or numbers, 
    anything that gets removed would be replaced by space.
    eg "wow.loved" would otherwise become "wowloved">> The space now helps to make the string as "wow loved"  :)
    
3) The third parameter is for specifying which row need to be processed in the algorithm, thats why we are running the loop for all rows
    
    
STEP 3 *********:
Pushing all the text to lowercase for uniformity

STEP 4 *********:
Splittinng each sentence to a word in a list.

STEP 5 *********:
Stemming is a process of getting the root word. for eg loved,loving,lovable >> love
love is the root word for all the above words and thus all these words reduce to 1 word love

STEP 6 *********:
Removing all the word that are connectors,prepositions etc. like "the","a","an","is","it"
Thats why we import nltk( Natural Language Tool Kit) : This librray contains a list of words that can be
 removed from the text.These list of words are called as stopwords.
 
STEP 7 *********:
Corpus or comments: it is the collection of text of same type.
'''


'''Step1 Lets remove all the numbers and alien characters and keep a-z and A-Z, also reolace them with space


Lets try all steps on the first data value :)

'''
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) 

'''Step2 Lets convert this string to all lower case for consistency'''

review = review.lower()

'''Step3 Splitting each comment and making an item of the list,
 split function work on str to make it a list'''

review = review.split()

'''Step4 Removing the stopwords, the words that are preopsitions etc..'''

review =[i for i in review if not i in set(stopwords.words('english'))]

'''
Let's try and see what the stopwords library holds

This is a complete list of words that have no weightage

'''

stop_list=stopwords.words('english')


'''Step5 Stemming where you keep the root word in the text'''
ps = PorterStemmer()
review = [ps.stem(j) for j in review if not j in set(stopwords.words('english'))]

'''Step6 joining all the words for making 1 string with 1 space'''
review = ' '.join(review)




'''
*************************************************************************************************
*************************************************************************************************
*************************************************************************************************
*************************************************************************************************
Lets now implemengt all the steps on entire data frame'''

comments = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    comments.append(review)
    
del i,review

'''
Improved code with no, not, nor in the text
'''

stop_list=stopwords.words('english')
stop = []
negative=["not","nor","no","ain","aren","aren't","isn","isn't","was","wasn't","didn't"]
stop = [i for i in stop_list if not i in negative]

comments = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stop]
    review = ' '.join(review)
    comments.append(review)
    
del i,review
'''
Creating the Bag of Words model: We would take all the distinct words from the comments( collection of text)
It would create a matrix for all unique words. The value would be 1 if the word exist, else a 0
This matrix with a lot of 0s is called as sparse matrix and the phenomena is called as sparsity

Tokenization: creating the column for each word from the comments

1. toarray() would convert list to a numpy array
2. We sue the max_features to pick the most frequent words, 
it would try and avoid the words that have just occured once
'''

'''Look the below code , it would give the count of distinct words or columns from the data'''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(comments).toarray()


''' Optimizing the above code for the top 1500 or you can even choose top 1000 most frequent words'''

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(comments).toarray()
y = dataset.iloc[:, 1].values

'''You can even analyze the excel, use .xlsx as .xls can have only 256 columns
Look at the excel

Every row is for every comment, the first row had the comment "wow love place" and thus total worda are 3

This is the count of 1 for the fisrt row.

We are just keeping the top 1500 most frequent words, similarly at the column you can know how many times that word came

'''

'''Exporting to excel, look at rows and columns counts, the row count is the number of words in first comment
The column count is the frequency of that word in all comments
'''
print(type(X))
df = pd.DataFrame(data=X)
df.to_excel(r'C:\Users\ami13\Desktop\X.xlsx')


''' Splitting the dataset into the Training set and Test set'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

''' Fitting Training set using knn with k=5'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

''' Predicting the Test set results'''
y_pred = classifier.predict(X_test)

'''
#****Evaluating the model based on confusion matrix score*************
# Making the Confusion Matrix
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''Validating the KNN  Model score and it is 58.5%'''
classifier.score(X_test, y_test)



    
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:04:33 2019

@author: rahim.chagani
"""

# Text Classifiation using NLP

# Importing the libraries
import numpy as np
import re
import pickle 
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

# Importing the dataset
reviews = load_files('txt_sentoken/')
X,y = reviews.data,reviews.target

# Pickling the dataset - Used to load huge data quickly.

with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)

# Unpickling dataset
X_in = open('X.pickle','rb')
y_in = open('y.pickle','rb')
X = pickle.load(X_in)
y = pickle.load(y_in)


# Creating the corpus
corpus = []
for i in range(0, 2000):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)    
    
# Creating the BOW model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()    
    
    
# Creating the Tf-Idf Model after bow of words model (BOW)
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


# Creating the Tf-Idf model directly
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()    
    
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 0)   
    
    
# Training the classifier - Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)    
    
# Testing model performance
sent_pred = classifier.predict(text_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)    

print(cm[0][0] + cm[1][1])
print(339/400)
    
    
# Saving our classifier
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
# Saving the Tf-Idf model
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)    
    
# Using our classifier
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
    
    
sample = ["You are an good person man, have a good life"]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))



# Twitter Sentiment Analysis using NLP - Live feed

# Install tweepy - pip install tweepy

# Importing the libraries
import tweepy
import re
import pickle

from tweepy import OAuthHandler

#Please change with your own consumer key, consumer secret, access token and access secret
# Initializing the keys
consumer_key = 'yoIwFkjZGYDa49aO16XqSNqcN'
consumer_secret = 'gl4LQOItV7Z1aFwNrlvaiKJ3t8o8h99blMIAmnmdHxYjzjRAxO' 
access_token = '624310916-E7fDF2IE8P6bfY1oVFglASf6F8RnxMd3vgSXFqnZ'
access_secret ='ID9JcoXHsDcKtvNcnmBGcCQhUlO0wmwAxBJ6LCesiUAas'


# Initializing the tokens
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
args = ['Trump'];
api = tweepy.API(auth,timeout=10)

# Fetching the tweets
list_tweets = []

query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent',geocode="22.1568,89.4332,500km").items(100):
        list_tweets.append(status.text)


# Initializing the tokens
#auth = OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_secret)
#args = ['facebook'];
#api = tweepy.API(auth,timeout=10)

# Fetching the tweets
#fblist_tweets = []

#query = args[0]
#if len(args) == 1:
#    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent').items(100):
#        fblist_tweets.append(status.text)


# Loading the vectorizer and classfier
with open('classifier.pickle','rb') as f:
    classifier = pickle.load(f)
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)    

clf.predict(tfidf.transform(['You are a bad person Mr.smart']))


total_pos = 0
total_neg = 0

# Preprocessing the tweets and predicting sentiment
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet) # Removing links in the start of the sentence
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet) # Removing links in the mid of the sentence
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet) # Removing links in the end of the sentence
    tweet = tweet.lower() # Chnage to loower case
    tweet = re.sub(r"that's","that is",tweet) # chnaging 's to is
    tweet = re.sub(r"there's","there is",tweet) # chnaging 's to is
    tweet = re.sub(r"what's","what is",tweet) # chnaging 's to is
    tweet = re.sub(r"where's","where is",tweet) # chnaging 's to is
    tweet = re.sub(r"it's","it is",tweet) # chnaging 's to is
    tweet = re.sub(r"who's","who is",tweet) # chnaging 's to is
    tweet = re.sub(r"i'm","i am",tweet) # chnaging 'm to am
    tweet = re.sub(r"she's","she is",tweet) # chnaging 's to is
    tweet = re.sub(r"he's","he is",tweet) # chnaging 's to is
    tweet = re.sub(r"they're","they are",tweet) # chnaging 're to are
    tweet = re.sub(r"who're","who are",tweet) # chnaging 're to are
    tweet = re.sub(r"ain't","am not",tweet) # chnaging ain't to am not
    tweet = re.sub(r"wouldn't","would not",tweet) # chnaging 't to not
    tweet = re.sub(r"shouldn't","should not",tweet) # chnaging 't to not
    tweet = re.sub(r"can't","can not",tweet) # chnaging 't to not
    tweet = re.sub(r"couldn't","could not",tweet) # chnaging 't to not
    tweet = re.sub(r"won't","will not",tweet)  # chnaging 't to not
    tweet = re.sub(r"\W"," ",tweet) # all punctuation
    tweet = re.sub(r"\d"," ",tweet) # all digits like #
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet) # Single character mid
    tweet = re.sub(r"\s+[a-z]$"," ",tweet) # Single character at end
    tweet = re.sub(r"^[a-z]\s+"," ",tweet) # Single character start
    tweet = re.sub(r"\s+"," ",tweet) # all extra space
    sent = classifier.predict(tfidf.transform([tweet]).toarray())
    #print(tweet,":",sent)
    if sent[0] == 1:
        total_pos += 1
    else:
        total_neg += 1


# Visualizing the results
import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive','Negative']
y_pos = np.arange(len(objects))

plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of Postive and Negative Tweets')

plt.show()






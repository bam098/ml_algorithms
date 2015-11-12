from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

print "\n"
print "Loading clean reviews..."
clean_train_reviews = pd.DataFrame.from_csv("data/labeledTrainDataClean.csv")
clean_utrain_reviews = pd.DataFrame.from_csv("data/unlabeledTrainDataClean.csv")

print "\n"
print "Fetching the vocabulary..."
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
data = pd.concat([clean_train_reviews['review'], clean_utrain_reviews['review']])
vectorizer.fit(np.array(data))
vocab = vectorizer.vocabulary_

print "\n"
print "Creating the bag of words..."
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000, vocabulary=vocab)
train_data_features = vectorizer.transform(np.array(clean_train_reviews['review']))
X = train_data_features.toarray()
y = clean_train_reviews['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_utrain = vectorizer.transform(np.array(clean_utrain_reviews['review'])).toarray()

print "\n"
print "Creating the model..."
clf = MultinomialNB()
clf.fit(X_train, y_train)

print "\n"
print "Testing the model..."
y_pred = []
for i in xrange(len(X_test)):
    pred = clf.predict(X_test[i])
    y_pred.append(pred)
acc = accuracy_score(y_test, np.asarray(y_pred))
print('Accuracy: %f' % acc)

print "\n"
print "Labeling unlabeled data with current model..."


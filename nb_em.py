from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def test_model(X, clf):
	y_pred = []
	for i in xrange(len(X)):
		pred = clf.predict(X[i])
		y_pred.append(pred)
	return accuracy_score(y_test, np.asarray(y_pred))

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
y = np.transpose(np.matrix(clean_train_reviews['sentiment']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_utrain = vectorizer.transform(np.array(clean_utrain_reviews['review'])).toarray()

print "\n"
print "Creating the model..."
clf = MultinomialNB()
clf.fit(X_train, np.ravel(y_train))
print "\n"
print "Testing the model..."
acc = test_model(X_test, clf)
print('Accuracy: %f' % acc)

epsilon = 0.000001
current_acc = -1.0
while np.abs(acc - current_acc) > epsilon:
	current_acc = acc

	print "\n"
	print "Labeling unlabeled data with current model..."
	y_pred = []
	for i in xrange(len(X_utrain)):
		pred = clf.predict(X_utrain[i])
		y_pred.append(pred)
	y_utrain = np.array(y_pred)	

	print "\n"
	print "Creating the model with labeled and unlabeled training data..."
	X_full = np.concatenate((X_train, X_utrain), axis=0)
	y_full = np.concatenate((y_train, y_utrain), axis=0)
	clf.fit(X_full, np.ravel(y_full))

	print "\n"
	print "Testing the model..."
	acc = test_model(X_test, clf)
	print('Accuracy: %f' % acc)
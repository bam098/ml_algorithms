import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
    
def train_naive_bayes(X, y):
    num_reviews = len(y)
    num_words = len(X[0])
    
    classes = np.unique(y)
    class_indices = np.array([y==a for a in classes])
    
    class_prior = np.matrix(([((len(y[a]))/float(num_reviews)) for a in class_indices]))
    class_log_prior = np.log(class_prior)

    review_lengths = np.sum(X, axis=1)
    word_class_counts = [np.sum(review_lengths[a]) for a in class_indices]
    
    words_by_class_sample = [X[a] for a in class_indices]  
    words_by_class = [np.sum(a, axis=0) for a in words_by_class_sample]
    word_probs = [np.divide(a+1, float(b+num_words)) for a,b in zip(words_by_class, word_class_counts)]
    word_log_probs = np.log(word_probs)

    params = np.hstack((np.transpose(class_log_prior), word_log_probs))
    
    return params
    
def predict_naive_bayes(x, params):
    x = np.matrix(x)
    intercept = np.ones([len(x), 1])
    x = np.hstack((intercept, x))
    scores = np.dot(x, np.transpose(params))
    pred = np.argmax(scores, axis=1)
    return np.asarray(pred).reshape(-1)
        
print "\n"
print "Loading clean reviews..."
clean_train_reviews = pd.DataFrame.from_csv("data/labeledTrainDataClean.csv")

print "\n"
print "Creating the bag of words..."
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(np.array(clean_train_reviews['review']))
X = train_data_features.toarray()
y = clean_train_reviews['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print "\n"
print "Creating the model..."
params = train_naive_bayes(X_train, y_train)

print "\n"
print "Testing the model..."
y_pred = []
for i in xrange(len(X_test)):
    pred = predict_naive_bayes(X_test[i], params)
    y_pred.append(pred)
acc = accuracy_score(y_test, np.asarray(y_pred))
print('Accuracy: %f' % acc)
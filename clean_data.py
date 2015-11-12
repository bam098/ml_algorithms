import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#import nltk
#nltk.download()

def review_to_words(raw_review):
    # Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # Convert to lower case , split into individual words
    words = letters_only.lower().split()

    # Define stop words
    stops = set(stopwords.words("english"))

    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # Join the words back into one string seperated by space
    return (" ".join(meaningful_words))

def clean_reviews(data):
    # Get the number of reviews
    num_reviews = data["review"].size

    # Init an empty list to hold the clean reviews
    clean_reviews = []

    # Loop over each review
    for i in xrange(0, num_reviews):
        if ((i+1)%1000 == 0):
            print "Review %d of %d" % (i+1, num_reviews)
        clean_reviews.append(review_to_words(data["review"][i]))
    
    column = pd.Series(clean_reviews)
    data['review'] = column
    return data

print "\n"
print "Loading labeled training set movie reviews..."
train = pd.read_csv('data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3, encoding="utf-8")

print "\n"
print "Cleaning and parsing the labeled training set..."
clean_train_reviews = clean_reviews(train)

print "\n"
print "Store clean labeled reviews..."
clean_train_reviews.to_csv("data/labeledTrainDataClean.csv")

print "\n"
print "Loading unlabeled training set movie reviews..."
utrain = pd.read_csv('data/unlabeledTrainData.tsv', header=0, delimiter='\t', quoting=3, encoding="utf-8")

print "\n"
print "Cleaning and parsing the unlabeled training set..."
clean_utrain_reviews = clean_reviews(utrain)

print "\n"
print "Store clean unlabeled labeled reviews..."
clean_utrain_reviews.to_csv("data/unlabeledTrainDataClean.csv")
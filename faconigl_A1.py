# COMP 4112 Introduction to Data Science
# Assignment 1, Classification
# Francesco Coniglione (st#1206780)

"""
Read in the TSV dataset. You can do this how you like; Python lists are totally acceptable but
you might have to convert to other formats for scikit-learn sometimes. If you want, you could
use a pandas dataframe or a numpy array.
"""

# Basic imports
import pandas as pd

# Other imports for analysis
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

data = pd.read_csv('email.tsv', sep='\t')

"""
Select/develop at least 6 features (generally going to be 6 Python functions). Yes, you could
use some of these values directly – go ahead and try them out! However, using a combination
of the data for a single feature might make it perform better. You can get creative here. There
are no necessarily wrong answ
"""

# Feature 1: Getting the number of characters in the email (in thousands)

def email_length(row):
    return row['num_char']

# Feature 2: Determine whether the email is sent to multiple recipients or has more than 10 cc's

def multiple_recipients(row):
    # Returns 1 for true, 0 for false
    return int(row['to_multiple'] or (row['cc'] > 10))

# Feature 3: Whether the email subject contains an exclamation mark or the word 'urgent'

def excited_subject(row):
    return int(row['exclaim_subj'] or row['urgent_subj'])

# Feature 4: Getting the number of attached files

def num_attachments(row):
    return int(row['attach'])

# Feature 5: Whether the email contains HTMl code

def contains_html(row):
    return int(row['format'])

# Feature 6: The number of times the email contains one or more of the following words or characters: 'dollar', 'inherit', 'viagra', 'password'

def spam_words(row):
    return sum([row['dollar'], row['inherit'], row['viagra'], row['password']])

"""
Train at least a KNN, Decision Tree/Random Forest(results will be similar since the Forest uses
Decision Trees), Naïve Bayes, and SVM classifier using the helper code below.

The following code was created using "The Basic Clickbait Classifier example code" as a reference.
"""

# Create the features and labels
# Reference: https://www.w3schools.com/python/pandas/ref_df_apply.asp
data['email_length'] = data.apply(email_length, axis=1)
data['multiple_recipients'] = data.apply(multiple_recipients, axis=1)
data['excited_subject'] = data.apply(excited_subject, axis=1)
data['num_attachments'] = data.apply(num_attachments, axis=1)
data['contains_html'] = data.apply(contains_html, axis=1)
data['spam_words'] = data.apply(spam_words, axis=1)

X = data[['email_length', 'multiple_recipients', 'excited_subject', 'num_attachments', 'contains_html', 'spam_words']]
y = data['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# KNN

classifierKNN = KNeighborsClassifier(n_neighbors=31)
classifierKNN.fit(X_train, y_train)
otherClassifierTestPred = classifierKNN.predict(X_test)
npYtest = np.array(y_test)
print("K-Nearest Neighbor" + "Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

# Decision Tree

classifierDTree = DecisionTreeClassifier()
classifierDTree.fit(X_train, y_train)
otherClassifierTestPred = classifierDTree.predict(X_test)
npYtest = np.array(y_test)
print("Decision Tree" + "Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

# Random Forest

classifierRndForest = RandomForestClassifier()
classifierRndForest.fit(X_train, y_train)
otherClassifierTestPred = classifierRndForest.predict(X_test)
npYtest = np.array(y_test)
print("Random Forest" + "Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

# Naive Bayes

classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)
otherClassifierTestPred = classifierNB.predict(X_test)
npYtest = np.array(y_test)
print("Gaussian NB" + "Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

# SVM

classifierSVM = svm.LinearSVC()
classifierSVM.fit(X_train, y_train)
otherClassifierTestPred = classifierSVM.predict(X_test)
npYtest = np.array(y_test)
print("SVM" + "Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))
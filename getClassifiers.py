
from __future__ import division
from sklearn import cross_validation
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import csv

# data from csv
data = list( csv.reader( open(  'data.csv', 'rU' ) ) )
data = np.array(data)

classLabel = data[:,0]
body =  data[:,4]
topic1 = data[:,5]

#Stop word removal, thresholding done here
myVec = CountVectorizer(min_df=1, ngram_range=(1, 1), stop_words="english")   #binary=true  #ngram_range=(2, 2)

#bag of words
unigrams = myVec.fit_transform(body)

#convert bag of words to array so that we can do cross fold
unigrams = unigrams.toarray()



#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# clf = svm.SVC()
# clf = RandomForestClassifier()
clf = GaussianNB()

def getMean(values):
    return np.mean(values)

def getStandardDev(values):
    return np.std(values)

macrec = []
acc = []
micrec = []
macprec = []
micprec = []


folds = KFold(len(classLabel), n_folds=10, indices=True)
for trainData, testData in folds:

    x_train, x_test, y_train, y_test = unigrams[trainData], unigrams[testData], classLabel[trainData], classLabel[testData]

    clf.fit(x_train, y_train)

    res = clf.predict(x_test)
    # Macro Precision
    macroPrec = metrics.precision_score(y_test, res, average='macro')
    macprec.append(macroPrec)

    # Accuracy
    accuracy = metrics.accuracy_score(y_test, res, normalize=True)
    acc.append(accuracy)

    # Micro Recall
    microRec = metrics.recall_score(y_test, res, average='micro')
    micrec.append(microRec)

    # Macro Recall
    macroRec = metrics.recall_score(y_test, res, average='macro')
    macrec.append(macroRec)

    # Micro Precision
    microPrec  = metrics.precision_score(y_test, res, average='micro')
    micprec.append(microPrec)

print getAverage(macprec)

print getMean(acc)

print getAverage(macrec)

print getAverage(micpre)

print getAverage(micrec)

print getStandardDev(acc)


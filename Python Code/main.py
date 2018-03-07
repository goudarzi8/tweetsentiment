import cPickle

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from time import time
import pandas as pn
from sklearn.feature_extraction.text import CountVectorizer

#Save Classifiers
from sklearn.externals import joblib

#Classifiers Libraries
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Ensemble Classifier Libraries
from sklearn.ensemble import VotingClassifier

#path
level = 'level2'
path = '/home/icebird/Downloads/project/'
dictionary = path + 'dictionary/' + level + '/'
datapath = path + 'data/'
print "main  " + level + "\n"
#load data
r_cols = ['sentiment','tweet']
if (level == 'level1'):
    data = pn.read_csv(datapath + 'training-data.csv',sep=',',names=r_cols,usecols=range(2),na_filter=True,header=None)
elif (level == 'level2'):
    data = pn.read_csv(datapath + 'clean-training-data.csv',sep=',',names=r_cols,usecols=range(2),na_filter=True,header=None)
elif (level == 'level3'):
    data = pn.read_csv(datapath + 'fixed-training-data.csv',sep=',',names=r_cols,usecols=range(2),na_filter=True,header=None)

data = data.dropna()
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['tweet'].values)
d = {'positive':1, 'negative':0}
targets = data['sentiment'].map(d)
totalPredictions = []

#benchmarking each classifier
def benchmark(clf,name) :
    #Loading the classifier
    print ('=' * 80)
    print name
    try :
        classifier = joblib.load( dictionary + name + '.pkl')
        train_time = 0
    except :
    
    #Training
        t0 = time()
        classifier = clf
        classifier.fit(counts, targets)
        train_time = time() - t0
        joblib.dump(classifier, dictionary + name + '.pkl') 
        print train_time
    
    #Testing
    t0 = time()
    predictions = classifier.predict(counts)
    test_time = time() - t0
    print test_time
    
    #Scoring
    score = metrics.accuracy_score(targets, predictions)
    print("accuracy:   %0.3f" % score)
    totalPredictions.append(predictions)
    return name, score, train_time, test_time

results = []

#Main Code

#Classifiers
clf1 = LogisticRegression()
clf2 = PassiveAggressiveClassifier()
clf3 = MultinomialNB(alpha=.01)
clf4 = BernoulliNB(alpha=.01)
clf5 = NearestCentroid()
clf6 = RidgeClassifier(tol=1e-2, solver="sag")
clf7 = Perceptron(n_iter=50)
clf8 = SGDClassifier(loss='hinge',alpha=.0001, n_iter=50,shuffle=True,penalty="l2",n_jobs=-1)
clf9 = SGDClassifier(loss='hinge',alpha=.0001, n_iter=50,shuffle=True,penalty="l1",n_jobs=-1)
clf10 = SGDClassifier(loss='hinge',alpha=.0001, n_iter=50,shuffle=True,penalty="elasticnet",n_jobs=-1)
clf11 = SGDClassifier(loss='log',alpha=.0001, n_iter=50,shuffle=True,penalty="l2",n_jobs=-1)
clf12 = SGDClassifier(loss='log',alpha=.0001, n_iter=50,shuffle=True,penalty="l1",n_jobs=-1)
clf13 = SGDClassifier(loss='log',alpha=.0001, n_iter=50,shuffle=True,penalty="elasticnet",n_jobs=-1)
clf14 = SGDClassifier(loss='modified_huber',alpha=.0001, n_iter=50,shuffle=True,penalty="l2",n_jobs=-1)
clf15 = SGDClassifier(loss='modified_huber',alpha=.0001, n_iter=50,shuffle=True,penalty="l1",n_jobs=-1)
clf16 = SGDClassifier(loss='modified_huber',alpha=.0001, n_iter=50,shuffle=True,penalty="elasticnet",n_jobs=-1)
clf17 = SGDClassifier(loss='squared_hinge',alpha=.0001, n_iter=50,shuffle=True,penalty="l2",n_jobs=-1)
clf18 = SGDClassifier(loss='squared_hinge',alpha=.0001, n_iter=50,shuffle=True,penalty="l1",n_jobs=-1)
clf19 = SGDClassifier(loss='squared_hinge',alpha=.0001, n_iter=50,shuffle=True,penalty="elasticnet",n_jobs=-1)
clf20 = SGDClassifier(loss='perceptron',alpha=.0001, n_iter=50,shuffle=True,penalty="l2",n_jobs=-1)
clf21 = SGDClassifier(loss='perceptron',alpha=.0001, n_iter=50,shuffle=True,penalty="l1",n_jobs=-1)
clf22 = SGDClassifier(loss='perceptron',alpha=.0001, n_iter=50,shuffle=True,penalty="elasticnet",n_jobs=-1)
clf23 = SGDClassifier(loss='squared_loss',alpha=.0001, n_iter=50,shuffle=True,penalty="l2",n_jobs=-1)
clf24 = SGDClassifier(loss='squared_loss',alpha=.0001, n_iter=50,shuffle=True,penalty="l1",n_jobs=-1)
clf25 = SGDClassifier(loss='squared_loss',alpha=.0001, n_iter=50,shuffle=True,penalty="elasticnet",n_jobs=-1)
clf26 = LinearSVC()
clf27 = LinearSVC(loss='squared_hinge', penalty="l2",dual=False, tol=1e-3)
clf28 = LinearSVC(loss='squared_hinge', penalty="l1",dual=False, tol=1e-3)
clf29 = DecisionTreeClassifier()
clf30 = RandomForestClassifier(n_estimators=100,n_jobs=-1)
#Ensemble Classifiers
eclf1 = VotingClassifier(estimators=[('clf2', clf2),('clf3',clf3),('clf4',clf4),('clf5',clf5),('clf7',clf7),('clf17',clf17),('clf18',clf18),('clf19',clf19),('clf20',clf20),('clf21',clf21),('clf23',clf23),('clf24',clf24),('clf25',clf25),('clf29',clf29),('clf30',clf30)], voting='hard',weights=[6,7,7,6,6,6,6,6,6,6,5,5,5,6,10])


for clf, name in (        
        (clf1,"LogisticRegression"),
        (clf2,"PassiveAggressiveClassifier"),
        (clf3, "MultinomialNB"),
        (clf4, "BernoulliNB"),
        (clf5, "NearestCentroid"),
        (clf6, "Ridge Classifier"),
        (clf7, "Perceptron"),
        (clf8, "SGD_hinge_l2"),
        (clf9, "SGD_hinge_l1"),
        (clf10, "SGD_hinge_elasticnet"),
        (clf11, "SGD_log_l2"),
        (clf12, "SGD_log_l1"),
        (clf13, "SGD_log_elasticnet"),
        (clf14, "SGD_modified_huber_l2"),
        (clf15, "SGD_modified_huber_l1"),
        (clf16, "SGD_modified_huber_elasticnet"),
        (clf17, "SGD_perceptron_l2"),
        (clf18, "SGD_perceptron_l1"),
        (clf19, "SGD_perceptron_elasticnet"),
        (clf20, "SGD_squared_hinge_l2"),
        (clf21, "SGD_squared_hinge_l1"),
        (clf22, "SGD_squared_hinge_elasticnet"),
        (clf23, "SGD_squared_loss_l2"),
	(clf24, "SGD_squared_loss_l1"),
	(clf25, "SGD_squared_loss_elasticnet"),
        (clf26, "LinearSVC"),
        (clf27, "LinearSVC_l2"),
        (clf28, "LinearSVC_l1"),
	(clf29,"DecisionTreeClassifier"),
	(eclf1, "Ensemble 1")):
    print('=' * 80)
    results.append(benchmark(clf,name))
    
#Saving the results    
with open(path + 'result/' + level + '/' + 'results.pkl', 'wb') as fid:
    cPickle.dump(results, fid)
  
# Drawing the graph
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

# Saving Correlation factor
corrnum =  pn.DataFrame(np.corrcoef(totalPredictions), results[0],results[0])

corrnum.to_excel(excel_writer = path + 'result/' + level + '/' + 'corrcoef.xlsx')

# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
class svm():
# Splitting the dataset into the Training set and Test set
  def svm(self,X,Y,split,kernel,c,g,d):
     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = split)

# Feature Scaling

     sc = StandardScaler()
     X_train = sc.fit_transform(X_train)
     X_test = sc.transform(X_test)

# Fitting SVM to the Training set

     classifier = SVC(kernel = kernel,probability=True,C=c,gamma=g,degree=d)
     classifier.fit(X_train, y_train)

# Predicting the Test set results
     y_pred = classifier.predict(X_test)
     pred = classifier.predict_proba(X_test)
     from sklearn.metrics import log_loss
     ll= log_loss(y_test, pred,normalize=False)
     acc=accuracy_score(y_test,y_pred)
# Making the Confusion Matrix
     precision, recall, fscore, support = score(y_test, y_pred)
     print 'll',ll
     print('precision: {}'.format(precision))
     print('recall: {}'.format(recall))
     print('fscore: {}'.format(fscore))
     print('support: {}'.format(support))
     cm = confusion_matrix(y_test, y_pred)
     print cm
     train_sizes=np.linspace(.1, 1.0, 5)
     cv = ShuffleSplit(n_splits=10, test_size=split)
     train_sizes, train_scores, test_scores = learning_curve(classifier, X, Y, cv=cv, n_jobs=1, train_sizes=train_sizes)
     train_scores_mean = np.mean(train_scores, axis=1)
     train_scores_std = np.std(train_scores, axis=1)
     test_scores_mean = np.mean(test_scores, axis=1)
     test_scores_std = np.std(test_scores, axis=1)
     #print train_scores#,train_sizes,test_scores
     #train_sizes, train_scores_svr, test_scores_svr = learning_curve(classifier, X,Y,train_sizes=np.linspace(0.1, 1, 10), cv=ShuffleSplit(n_splits=100, test_size=split, random_state=0),n_jobs=1)
     #train_scores_svr = np.mean(train_scores_svr, axis=1)
     #test_scores_svr = np.mean(test_scores_svr, axis=1)
     #print train_scores_svr
     return y_pred,cm,train_sizes,train_scores_mean,test_scores_mean,precision,recall,fscore,support,ll,acc

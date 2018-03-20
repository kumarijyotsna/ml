# Cross Validation Classification Accuracy
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.utils import shuffle
from sklearn.model_selection import validation_curve
from sklearn.metrics import precision_recall_fscore_support as score
#from sklearn.metrics import precision_score,recall_score,accuracy_score, f1_score
#import snips as snp
#snp.prettyplot(matplotlib) 
class logistic_regression():
  def lr(self,X,Y,c,split):


    lab_enc = preprocessing.LabelEncoder()
    Y = lab_enc.fit_transform(Y)
    print Y
    X, Y = shuffle(X, Y)
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = LogisticRegression(C=c)
    #scoring = 'accuracy'
    #results1 = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    #print("Accuracy: %.3f (%.3f)") % (results1.mean(), results1.std())
    
    
    #scoring = 'roc_auc'
    #results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    #print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=split, random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    matrix = confusion_matrix(Y_test, predicted)
    print(matrix)
    
    pred = model.predict_proba(X_test)
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(Y_test,predicted)
    precision, recall, fscore, support = score(Y_test, predicted)
    
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print precision[0]
    from sklearn.metrics import log_loss
    results2= log_loss(Y_test, pred,normalize=False)
    param_range = np.logspace(-6, -1, 5)
    train_sizes, train_scores, valid_scores = learning_curve(model, X, Y, train_sizes=[0.1, 0.33, 0.55, 0.78, 1.], cv=5)
    train_scores = np.mean(train_scores, axis=1)
    valid_scores = np.mean(valid_scores, axis=1)
    '''train_scores, test_scores = validation_curve(model, X, Y, "alpha", np.logspace(-7, 3, 3))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)'''
    #fig, ax = plt.subplots()
    #ax.plot(train_sizes, train_scores, linestyle="--", color="r", label="training error")
    #ax.plot(train_sizes, valid_scores, linestyle="-", color="b", label="cv error")

    #snp.labs("Training Set Size", "Score (4-Fold CV avg)", "LC with High Bias")
    #ax.legend(loc="lower right")
    print train_scores,valid_scores,train_sizes
    return results2,matrix,train_scores,valid_scores,train_sizes,acc,precision,recall,fscore,support#,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std


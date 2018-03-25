import numpy as np
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
class svr():
   '''def tts(self,X,Y):
      X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=split, random_state=0)
      sc = StandardScaler()
      X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)
      return X_train,X_test,Y_train,Y_test'''
    
   def svr(self,X,Y,C,E,G,degree,split,kernel):
       #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=split, random_state=0)
       sc = StandardScaler()
       print X,Y,degree
       X = sc.fit_transform(X)
       #y=sc.fit_transform(Y)
       
       svr = SVR(kernel=kernel,C=C,epsilon=E,gamma=G,degree=degree)
       y_pred = svr.fit(X, Y).predict(X)
       accr=svr.score(X,Y)
       #y_pred = sc.inverse_transform(y_pred)
       '''y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
       y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
       y_sig = svr_sig.fit(X_train, y_train).predict(X_test)'''
       train_sizes, train_scores_svr, test_scores_svr = learning_curve(svr, X,Y,train_sizes=np.linspace(0.1, 1, 10),scoring="neg_mean_squared_error", cv=10)
       train_scores_svr = np.mean(train_scores_svr, axis=1)
       test_scores_svr = np.mean(test_scores_svr, axis=1)
       print svr.get_params
       return y_pred,train_sizes,train_scores_svr,test_scores_svr,accr

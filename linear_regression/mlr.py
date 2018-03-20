import numpy as np
import pandas as pd

from numpy import double

from sklearn.preprocessing import StandardScaler
class mlr:
        def find(self,data):
          minmax=[]
          for i in range(len(data[0])):
            r=[row[i] for row in data]
            min_val=min(r)
            max_val=max(r)
          if (min_val==max_val):
             max_val=1
          minmax.append([min_val,max_val])
          return minmax
        def standard(self,X):
           scaler = StandardScaler().fit(X)
           rescaledX = scaler.transform(X)
# summarize transformed data
           np.set_printoptions(precision=3)
           print(rescaledX)
           return rescaledX  
        def normalizer(self,data,minmax): 
          #print data
          print minmax[0][0],minmax[0][1]
          #print data
          print len(data),len(data[0])
          for i in range(0,len(data)):
            for j in range(0,len(data[0])):
              #print i,j
              data[i][j]=(data[i][j]-minmax[0][0])/(minmax[0][1]-minmax[0][0])
              print data[i][j] 
          return data
        def cost_function(self,X, Y, B):
	    m = len(Y)
            
	    J = np.double(np.sum((X.dot(B) - Y) ** 2)/(2 * m))
	    return J
	
	def gradient_descent(self,X, Y, B, alpha, iterations):
	    cost_history = [0] * iterations
	    m = len(Y)
	    
	    for iteration in range(iterations):
		# Hypothesis Values
		h = X.dot(B)
		# Difference b/w Hypothesis and Actual Y
		loss = h - Y
		# Gradient Calculation
		gradient = X.T.dot(loss) / m
		# Changing Values of B using Gradient
		B = B - alpha * gradient
		# New Cost Value
		cost = self.cost_function(X, Y, B)
		cost_history[iteration] = cost
		
	    return B, cost_history

	


	# Model Evaluation - RMSE
	def rmse(self,Y, Y_pred):
	    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
	    return rmse

	# Model Evaluation - R2 Score
	def r2_score(self,Y, Y_pred):
	    mean_y = np.mean(Y)
	    ss_tot = sum((Y - mean_y) ** 2)
	    ss_res = sum((Y - Y_pred) ** 2)
	    r2 = 1 - (ss_res / ss_tot)
	    return r2

  

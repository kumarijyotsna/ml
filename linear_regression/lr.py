import numpy as np
import random
class lr():

  def lr(self,X,Y):
        # Mean X and Y
	mean_x = np.mean(X)
	mean_y = np.mean(Y)

	# Total number of values
	m = len(X)

	# Using the formula to calculate b1 and b2
	
        numer = 0
	denom = 0
	for i in range(m):
	    numer += (X[i] - mean_x) * (Y[i] - mean_y)
	    denom += (X[i] - mean_x) ** 2
	b1 = numer / denom
	b0 = mean_y - (b1 * mean_x)
        print "b1",b1,b0
	# Print coefficients
	#print(b1, b0)
	max_x = np.max(X) + 100
	min_x = np.min(X) - 100

	# Calculating line values x and y
	x = np.linspace(min_x, max_x, 1000)
	y = b0 + b1 * x
	# Calculating Root Mean Squares Error
	rmse = 0
        Y_pred=[]
	for i in range(m):
	    y_pred = b0 + b1 * X[i]
            Y_pred.append(y_pred)
	    rmse += (Y[i] - y_pred) ** 2
	rmse = np.sqrt(rmse/m)
	#print(rmse)
        #print "y",y_pred,rmse,y
        return Y_pred,rmse,b1,b0
  def test_lr(self,X,Y,b1,b0):
        m = len(X)
        max_x = np.max(X) + 100
	min_x = np.min(X) - 100

	# Calculating line values x and y
	x = np.linspace(min_x, max_x, 1000)
	y = b0 + b1 * x
	# Calculating Root Mean Squares Error
	rmse = 0
        Y_pred=[]
	for i in range(m):
	    y_pred = b0 + b1 * X[i]
            Y_pred.append(y_pred)
	    rmse += (Y[i] - y_pred) ** 2
	rmse = np.sqrt(rmse/m)
	#print(rmse)
        print "y",Y_pred[1:5],rmse
        return Y_pred,rmse

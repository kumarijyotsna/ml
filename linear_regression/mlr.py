import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

class mlr:
        def cost_function(self,X, Y, B):
	    m = len(Y)
	    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
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

  

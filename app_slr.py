from flask import Flask
from flask import render_template,make_response
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import csv
import pandas as pd
import numpy as np
from linear_regression import lr
from linear_regression import *
from linear_regression import file_import
lr=lr()
m_lr=mlr()
fi=file_import()
app = Flask(__name__)
UPLOAD_FOLDER = '/home/simmi'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def plot_data(x):
   j=[]
  
   for i in range(0,len(x)):
      j.append(x[i])
     
   return j
header=[]
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')
@app.route('/slr', methods=['GET', 'POST'])
def slr():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        fi.file_path(file.filename)
        if file and fi.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return filepath
            data = pd.read_csv(filepath)
	X = data.iloc[:,0].values
	Y = data.iloc[:,1].values
        
	Y_pred,rmse=lr.lr(X,Y)
        #print rmse
        #print Y_pred   
	  
        x1=plot_data(X)
        y1=plot_data(Y)
        #print y1,x1
        y1_pred=plot_data(Y_pred)
        return render_template('result.html',x=x1,y=y1,rmse=rmse,y_pred=y1_pred)
            
    return render_template('file_render.html')

@app.route('/mlr',methods=['GET','POST'])
def mmlr():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        fi.file_path(file.filename)
        if file and fi.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           
            df = pd.read_csv(filepath)
            df_n=pd.read_csv(filepath)
	    Y = df.iloc[:,-1].values
           
           
            df.drop(df.columns[len(df.columns)-1],axis=1,inplace=True)
            
            global header
            with open(filepath,'r') as f:
              reader=csv.reader(f)
              header.append(reader.next())
            header=np.array(header[0])
            for i in range(0,len(df_n.columns)):
            
              if(df_n.dtypes[i]== object):
               
                df_ne=pd.get_dummies(df_n[header[i]],drop_first=True,prefix=[str(i)],prefix_sep='_')
              
                df.drop(header[i],axis=1,inplace=True)
                df=df.join(pd.DataFrame(df_ne))
               
            #print  len(df.columns)
            header=[]
            sf=df.values.tolist()
            #print df.columns
            X = df.iloc[:,0:len(df.columns)].values
            #print X,Y
            m = len(X)
            
	    x0 = np.ones(m)
            #print x0
	    X =  np.append(arr=np.ones((m,1)).astype(int),values=X,axis=1)
	    #print X
	# Initial Coefficients
	    B = np.zeros(len(df.columns)+1)
            #print B
	    alpha = 0.0001

            inital_cost = m_lr.cost_function(X, Y, B)
	    print(inital_cost)
            newB, cost_history = m_lr.gradient_descent(X, Y, B, alpha, 100000)
            
	# New Values of B
	    print(newB)

	# Final Cost of new B
	    print(cost_history[-1]) 
            Y_pred = X.dot(newB)
            print(m_lr.rmse(Y, Y_pred))
            print(m_lr.r2_score(Y, Y_pred))

        return render_template('result_m.html',rmse=m_lr.rmse(Y, Y_pred))
   return render_template('file_render.html')
   
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)


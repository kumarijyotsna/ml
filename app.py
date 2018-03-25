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
import pymysql.cursors
from flask import session
import json
lr=lr()
m_lr=mlr()
fi=file_import()
lor=logistic_regression()
sv_r=svr()
sv_c=svm()
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


@app.route('/file',methods=['GET', 'POST'])
def index():  
  types=[]
  header=[]
  sample=[]

  if request.method == 'POST':
     if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
     file = request.files['file']
     #split=request.form['split']
     #print split
     if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
     fi.file_path(file.filename)
     if file and fi.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      
            df = pd.read_csv(filepath)
     #header1=[]     
     splt1=request.form['splt']
     print splt1
     session['split']=splt1
     with open(filepath,'r') as f:
              reader=csv.reader(f)
              header.append(reader.next())
              sample=next(reader)
     print header[0]
     '''for x in header[0]:
         pattern = re.compile(r'\s+')
         x = re.sub(pattern, '', x)
         print x
         header1.append(x)'''
     header1=np.array(header[0])
     print header1
     print df.dtypes
     session['var']=filepath
     #print (df.dtypes)
     for i in range(0, len(df.columns)):
          #print(df.dtypes[i])
            
          if(df.dtypes[i]=='int64'):
             types.append('int')
          if(df.dtypes[i]=='object'):
             types.append('varchar(100)')
          if(df.dtypes[i]=='float64'):
             types.append('real')
     print len(types)
     print sample
     return render_template("choice.html",header=zip(header1,types,sample),file_n=file.filename)
  
  return render_template('file_render.html')




@app.route('/slr', methods=['POST','GET'])
def slr():
 if request.method =='POST' or request.method=='GET':
   #if request.method =='POST':
   #n_data=request.form['text']
   #n_data1=request.form['text1']
   #print(n_data)
   #else:
    #n_data=request.form.get('text')
    #print (n_data)
   n_data=request.args.get('text')
   #n_data=request.form.get('text')
   print n_data
   import sqlite3
   
   con = sqlite3.connect('database.db')
   varf=session.get('var',None)
   split=session.get('split', None)
   print (varf)
   
   split=float(split)
 
   print "type",type(split)
   data_f=varf.split('/')
   print data_f[-1]
   for x in data_f:
      df=x.split('.')
   sql = "DROP TABLE IF EXISTS %s" % df[0]
   con.execute(sql)
   query='CREATE TABLE '+ df[0]+'('+n_data+')'
   con.execute(query)
   data=n_data.split(',')
   var1=[]
   #print data
   for x in data:
        d=x.split(' ')
        print d
        for y in d:
           #print y 
           if(y!='int' and y!='varchar(100)' and y!='real'):
              var1.append(str(y))
   print len(var1)  
   var=tuple(var1)
   print var
   to_db=[]
   with open(varf,'rb') as fin:
      dr=csv.DictReader(fin)
      for i in dr:
        to_db.append([(i[n]) for n in var])
        #print to_db
   print len(to_db)
   q1=[]
   for i in range(0,len(var1)):
       q1.append('?')
 
   q=",".join(q1)
   #print q
   quer="INSERT INTO "+df[0]+" "+ str(var)+" VALUES("+q+");"
   #print quer
   con.executemany("INSERT INTO "+df[0]+" "+ str(var)+" VALUES("+(q)+");",to_db)
   con.commit()
   con.execute("PRAGMA busy_timeout = 30000")
   con.close() 
   conn = sqlite3.connect("database.db")
   data = pd.read_sql_query("select * from "+ df[0], conn)
   #print data
   #print len(data)
   data = data.iloc[np.random.permutation(len(data))]
   data= data.reset_index(drop=True)
   split_point = int(len(data)*(split))
   train, test = data[:split_point], data[split_point:]
   #print len(train)
   #print len(test)
   #print train.head()
   #print test.head()
   #train,test=lr.split_dataset(data, split)
   X_train = train.iloc[:,0].values
   Y_train = train.iloc[:,1].values
   X_test=test.iloc[:,0].values
   Y_test=test.iloc[:,1].values
   #print Y_test[1:5]
   Y_pred_train,rmse_train,m,b=lr.lr(X_train,Y_train)
   print "m",m,b
   print Y_pred_train[1:5]
   Y_pred_test,rmse_test=lr.test_lr(X_test,Y_test,m,b)
        #print rmse
   print Y_pred_test  
   print "test",Y_pred_test[1:5],Y_test[1:5]
   x1=plot_data(X_train)
   y1=plot_data(Y_train)
   #print "train",X_train,len(x1)
   print x1
   print X_train
        #print y1,x1
   y1_pred=plot_data(Y_pred_train)
   x2=plot_data(X_test)
   y2=plot_data(Y_test)
        #print y1,x1
   y2_pred=plot_data(Y_pred_test)
   #print "test",len(X_test),len(x2)
   #print x2
   session['accr']=rmse_train
   session['accr1']=rmse_test
   session['n_data']=n_data
   
   return render_template('result.html',x=x1,y=y1,y_pred=y1_pred,x1=x2,y1=y2,y1_pred=y2_pred ,n_data=n_data)
  #else :
   #return "get"
@app.route('/accuracy',methods=['POST','GET'])
def accuracy():
    accr=session.get('accr',None)     
    accr1=session.get('accr1',None)
    n_data=session.get('n_data',None)
    return render_template('accr.html',rmse1=accr,rmse2=accr1,n_data=n_data)
@app.route('/file1',methods=['GET', 'POST'])
def index1():  
  types=[]
  header=[]
  sample=[]
  if request.method == 'POST':
     if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
     file = request.files['file']
     if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
     fi.file_path(file.filename)
     if file and fi.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           
            df = pd.read_csv(filepath)
     #header1=[]     
     with open(filepath,'r') as f:
              reader=csv.reader(f)
              header.append(reader.next())
              sample=next(reader)
     print header[0]
     '''for x in header[0]:
         pattern = re.compile(r'\s+')
         x = re.sub(pattern, '', x)
         print x
         header1.append(x)'''
     header1=np.array(header[0])
     print header1
     print df.dtypes
     session['var']=filepath
     #print (df.dtypes)
     for i in range(0, len(df.columns)):
          #print(df.dtypes[i])
            
          if(df.dtypes[i]=='int64'):
             types.append('int')
          if(df.dtypes[i]=='object'):
             types.append('varchar(100)')
          if(df.dtypes[i]=='float64'):
             types.append('real')
     print len(types)
     print sample
     alpha = float(request.form["alpha"])
     iterations=int(request.form["iterations"])
     session['alpha']=alpha
     session['iterations']=iterations
     return render_template("choice1.html",header=zip(header1,types,sample),file_n=file.filename)
  
  return render_template('file_render_m.html')


@app.route('/mlr',methods=['POST','GET'])
def mmlr():
   n_data=request.form['text']
   
   print(n_data)
   
   import sqlite3
   
   con = sqlite3.connect('database.db')
   varf=session.get('var',None)
   print (varf)
   data_f=varf.split('/')
   print data_f[-1]
   for x in data_f:
      dfx=x.split('.')
   sql = "DROP TABLE IF EXISTS %s" % dfx[0]
   con.execute(sql)
   query='CREATE TABLE '+ dfx[0]+'('+n_data+')'
   con.execute(query)
   data=n_data.split(',')
   var1=[]
   #print data
   for x in data:
        d=x.split(' ')
        print d
        for y in d:
           #print y 
           if(y!='int' and y!='varchar(100)' and y!='real'):
              var1.append(str(y))
   print len(var1)  
   var=tuple(var1)
   print var
   to_db=[]
   with open(varf,'rb') as fin:
      dr=csv.DictReader(fin)
      for i in dr:
        to_db.append([(i[n]) for n in var])
        #print to_db
   print len(to_db)
   q1=[]
   for i in range(0,len(var1)):
       q1.append('?')
 
   q=",".join(q1)
   print q
   quer="INSERT INTO "+dfx[0]+" "+ str(var)+" VALUES("+q+");"
   print quer
   con.executemany("INSERT INTO "+dfx[0]+" "+ str(var)+" VALUES("+(q)+");",to_db)
   con.commit()
   con.execute("PRAGMA busy_timeout = 30000")
   con.close() 
   conn = sqlite3.connect("database.db")
   df= pd.read_sql_query("select * from "+ dfx[0], conn)
   conn.close()
   conn=sqlite3.connect("database.db")
   df_n= pd.read_sql_query("select * from "+ dfx[0], conn)
   #print df[0:5]
   #print df_n[0:5]
   Y = df.iloc[:,-1].values
           
   print df_n.dtypes
   print len(df_n.columns)      
   df.drop(df.columns[len(df.columns)-1],axis=1,inplace=True)
   
   
   for j in range(0,len(df_n.columns)):        
              #print df_n.dtypes[j]
              #print j
              if(df_n.dtypes[j]== object):
                #print j
                df_ne=pd.get_dummies(df_n[var1[j]],drop_first=True,prefix=[str(j)],prefix_sep='_')
                #print df_ne
                df.drop(var1[j],axis=1,inplace=True)
                df=df.join(pd.DataFrame(df_ne))
                #print df.head()     
            #print  len(df.columns)
   header=[]
      
   sf=df.values.tolist()
            #print df.columns
   X = df.iloc[:,0:len(df.columns)].values
   minmax=m_lr.find(X)
   X=m_lr.standard(X)
   #X=m_lr.normalizer(X,minmax)
   #print X,minmax
   #print X,Y
   m = len(X)
   #print len(X[0])
   #print X[0]
   #print X[1]
   x0 = np.ones(m)
            #print x0
   X =  np.append(arr=np.ones((m,1)).astype(int),values=X,axis=1)
	    #print X
	# Initial Coefficients
   B = np.zeros(len(df.columns)+1)        
            #print B
   alpha = float(session.get('alpha',None))
   iterations=int(session.get('iterations',None))         
   print (iterations)        
   inital_cost = m_lr.cost_function(X, Y, B)         
   print (inital_cost)        
   newB, cost_history = m_lr.gradient_descent(X, Y, B, alpha, iterations)         
   print(newB)
         
	# New Values of B
            
	# Final Cost of new B
   print (cost_history[-1])          
            
   Y_pred = X.dot(newB)        
   print(m_lr.rmse(Y, Y_pred))
   print(m_lr.r2_score(Y, Y_pred))

   return render_template('result_m.html',rmse=m_lr.rmse(Y,Y_pred))
            
@app.route('/file2',methods=['GET', 'POST'])
def index2():  
  types=[]
  header=[]
  sample=[]

  if request.method == 'POST':
     if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
     file = request.files['file']
     #split=request.form['split']
     #print split
     if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
     fi.file_path(file.filename)
     if file and fi.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      
            df = pd.read_csv(filepath)
     #header1=[]     
     splt1=request.form['splt']
     c=request.form['c']
     c=float(c)
     splt1=float(splt1)
     print splt1
     session['split']=splt1
     session['c']=c
     with open(filepath,'r') as f:
              reader=csv.reader(f)
              header.append(reader.next())
              sample=next(reader)
     print header[0]
     '''for x in header[0]:
         pattern = re.compile(r'\s+')
         x = re.sub(pattern, '', x)
         print x
         header1.append(x)'''
     header1=np.array(header[0])
     print header1
     print df.dtypes
     session['var']=filepath
     #print (df.dtypes)
     for i in range(0, len(df.columns)):
          #print(df.dtypes[i])
            
          if(df.dtypes[i]=='int64'):
             types.append('int')
          if(df.dtypes[i]=='object'):
             types.append('varchar(100)')
          if(df.dtypes[i]=='float64'):
             types.append('real')
     print len(types)
     print sample
     return render_template("choice3.html",header=zip(header1,types,sample),file_n=file.filename)
  
  return render_template('file_render_lr.html')


@app.route('/lo_r',methods=['POST','GET'])
def lo_r():
  if request.method =='POST' or request.method=='GET':
   n_data=request.form['text']
   #n_data=request.args.get('text')
   
   print(n_data)
   
   import sqlite3
   
   con = sqlite3.connect('database.db')
   varf=session.get('var',None)
   split=session.get('split',None)
   c=session.get('c',None)
   print (varf),c
   data_f=varf.split('/')
   print data_f[-1]
   for x in data_f:
      dfx=x.split('.')
   sql = "DROP TABLE IF EXISTS %s" % dfx[0]
   con.execute(sql)
   query='CREATE TABLE '+ dfx[0]+'('+n_data+')'
   con.execute(query)
   data=n_data.split(',')
   var1=[]
   #print data
   for x in data:
        d=x.split(' ')
        print d
        for y in d:
           #print y 
           if(y!='int' and y!='varchar(100)' and y!='real'):
              var1.append(str(y))
   print len(var1)  
   var=tuple(var1)
   print var
   to_db=[]
   with open(varf,'rb') as fin:
      dr=csv.DictReader(fin)
      for i in dr:
        to_db.append([(i[n]) for n in var])
        #print to_db
   print len(to_db)
   q1=[]
   for i in range(0,len(var1)):
       q1.append('?')
 
   q=",".join(q1)
   print q
   quer="INSERT INTO "+dfx[0]+" "+ str(var)+" VALUES("+q+");"
   print quer
   con.executemany("INSERT INTO "+dfx[0]+" "+ str(var)+" VALUES("+(q)+");",to_db)
   con.commit()
   con.execute("PRAGMA busy_timeout = 30000")
   con.close() 
   conn = sqlite3.connect("database.db")
   df= pd.read_sql_query("select * from "+ dfx[0], conn)
   conn.close()
   conn=sqlite3.connect("database.db")
   df_n= pd.read_sql_query("select * from "+ dfx[0], conn)
   #print df[0:5]
   #print df_n[0:5]
   Y = df.iloc[:,-1].values
           
   print df_n.dtypes
   print len(df_n.columns)      
   df.drop(df.columns[len(df.columns)-1],axis=1,inplace=True)
   
   
   for j in range(0,len(df_n.columns)):        
              #print df_n.dtypes[j]
              #print j
              if(df_n.dtypes[j]== object):
                #print j
                df_ne=pd.get_dummies(df_n[var1[j]],drop_first=True,prefix=[str(j)],prefix_sep='_')
                #print df_ne
                df.drop(var1[j],axis=1,inplace=True)
                df=df.join(pd.DataFrame(df_ne))
                #print df.head()     
            #print  len(df.columns)
   header=[]
      
   sf=df.values.tolist()
            #print df.columns
   X = df.iloc[:,0:len(df.columns)].values

   results2,matrix,report,trn_s,cv_s,t_size,accr,precision,recall,fscore,support=lor.lr(X,Y,c,split)
   #print matrix[0]
   trn_s=list(trn_s)
   cv_s=list(cv_s)
   t_size=list(t_size)
   #print tr_cvm
   lm= len(matrix)
   print type(report)
   #session['lm']=lm
   session['accr']=accr
   session['results2']=results2
   session['matrix']=matrix.tolist()
   session['report']=report
   #session['n_data']=n_data
   session['trn_s']=trn_s
   session['cv_s']=cv_s
   session['t_size']=t_size
   
   session['p']=precision.tolist()
   session['r']=recall.tolist()
   session['f']=fscore.tolist()
   session['s']=support.tolist()
   return render_template("result3.html",classification=zip(precision,recall,fscore,support))
@app.route('/accuracy_lo',methods=['POST','GET'])
def accuracy_lo():
    accr2=session.get('accr',None)
    #n_data=session.get('n_data',None)
    return render_template('accr_lo.html',accuracy=accr2)
@app.route('/loss_lo',methods=['POST','GET'])
def loss_lo():
    accr1=session.get('results2',None)     
   
    #n_data=session.get('n_data',None)
    return render_template('loss_lo.html',results2_m=accr1)
@app.route('/matrix_lo',methods=['POST','GET'])
def matrix_lo():
    lm=session.get('lm',None)
    matrix=session.get('matrix',None)     
    #accr2=session.get('results1_s',None)
    #n_data=session.get('n_data',None)
    return render_template('matrix_lo.html',matrix=matrix,lm=lm)
@app.route('/report_lo',methods=['POST','GET'])
def report_lo():
    p=session.get('p',None)
    r=session.get('r',None)
    f=session.get('f',None)     
    s=session.get('s',None)
    #accr2=session.get('results1_s',None)
    #n_data=session.get('n_data',None)
    return render_template('report_lo.html',classification=zip(p,r,f,s))
@app.route('/graph_lo',methods=['POST','GET'])
def graph():
       
    trn_s=session.get('trn_s',None)
    cv_s=session.get('cv_s',None)
    t_size=session.get('t_size',None)

    return render_template("graph_lo.html",trn_s=trn_s,cv_s=cv_s,t_size=t_size)


@app.route('/file3',methods=['GET', 'POST'])
def index3():  
  types=[]
  header=[]
  sample=[]
  if request.method == 'POST':
     if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
     file = request.files['file']
     if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
     fi.file_path(file.filename)
     if file and fi.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           
            df = pd.read_csv(filepath)
     #header1=[]     
     with open(filepath,'r') as f:
              reader=csv.reader(f)
              header.append(reader.next())
              sample=next(reader)
     print header[0]
     '''for x in header[0]:
         pattern = re.compile(r'\s+')
         x = re.sub(pattern, '', x)
         print x
         header1.append(x)'''
     header1=np.array(header[0])
     print header1
     print df.dtypes
     session['var']=filepath
     #print (df.dtypes)
     for i in range(0, len(df.columns)):
          #print(df.dtypes[i])
            
          if(df.dtypes[i]=='int64'):
             types.append('int')
          if(df.dtypes[i]=='object'):
             types.append('varchar(100)')
          if(df.dtypes[i]=='float64'):
             types.append('real')
     print len(types)
     print sample
     c = float(request.form["c"])
     epsilon=float(request.form["epsilon"])
     gamma=float(request.form["gamma"])
     degree=int(request.form["degree"])
     kernel=str(request.form["kernel"])
     split=float(request.form["split"])
     session['c']=c #c must be graeter than 1 in case of noisy data
     session['epsilon']=epsilon
     session['gamma']=gamma
     session['degree']=degree #used in case of poly kernel
     session['kernel']=kernel
     session['split']=split
     print c,epsilon,gamma,degree,kernel,split
     return render_template("choice4.html",header=zip(header1,types,sample),file_n=file.filename)
  
  return render_template('file_render_svr.html')
@app.route('/svr',methods=['POST','GET'])
def svr():
  if request.method =='POST' or request.method=='GET':
   n_data=request.form['text']
   #n_data=request.args.get('text')
   
   print(n_data)
   
   import sqlite3
   
   con = sqlite3.connect('database.db')
   varf=session.get('var',None)
   s=session.get('split',None)
   c=session.get('c',None)
   e=session.get('epsilon',None)
   g=session.get('gamma',None)
   degree=session.get('degree',None)
   #degree=int(d)
   print type(degree)
   k=session.get('kernel',None)
   print (varf),c,e,g,degree,k
   data_f=varf.split('/')
   print data_f[-1]
   for x in data_f:
      dfx=x.split('.')
   sql = "DROP TABLE IF EXISTS %s" % dfx[0]
   con.execute(sql)
   query='CREATE TABLE '+ dfx[0]+'('+n_data+')'
   con.execute(query)
   data=n_data.split(',')
   var1=[]
   #print data
   for x in data:
        d=x.split(' ')
        print d
        for y in d:
           #print y 
           if(y!='int' and y!='varchar(100)' and y!='real'):
              var1.append(str(y))
   print len(var1)  
   var=tuple(var1)
   print var
   to_db=[]
   with open(varf,'rb') as fin:
      dr=csv.DictReader(fin)
      for i in dr:
        to_db.append([(i[n]) for n in var])
        #print to_db
   print len(to_db)
   q1=[]
   for i in range(0,len(var1)):
       q1.append('?')
 
   q=",".join(q1)
   print q
   quer="INSERT INTO "+dfx[0]+" "+ str(var)+" VALUES("+q+");"
   print quer
   con.executemany("INSERT INTO "+dfx[0]+" "+ str(var)+" VALUES("+(q)+");",to_db)
   con.commit()
   con.execute("PRAGMA busy_timeout = 30000")
   con.close() 
   conn = sqlite3.connect("database.db")
   df= pd.read_sql_query("select * from "+ dfx[0], conn)
   conn.close()
   conn=sqlite3.connect("database.db")
   df_n= pd.read_sql_query("select * from "+ dfx[0], conn)
   #print df[0:5]
   #print df_n[0:5]
   Y = df.iloc[:,-1].values
           
   print df_n.dtypes
   print len(df_n.columns)      
   df.drop(df.columns[len(df.columns)-1],axis=1,inplace=True)
   
   
   for j in range(0,len(df_n.columns)):        
              #print df_n.dtypes[j]
              #print j
              if(df_n.dtypes[j]== object):
                #print j
                df_ne=pd.get_dummies(df_n[var1[j]],drop_first=True,prefix=[str(j)],prefix_sep='_')
                #print df_ne
                df.drop(var1[j],axis=1,inplace=True)
                df=df.join(pd.DataFrame(df_ne))
                #print df.head()     
            #print  len(df.columns)
   header=[]
      
   sf=df.values.tolist()
   print df.columns
   X = df.iloc[:,0:len(df.columns)].values
   x1=X.flatten()
   x1=list(x1)
   y_pred,train_sizes,train_scores_svr,test_scores_svr,accr=sv_r.svr(X,Y,c,e,g,degree,s,k)
   #print matrix[0]
   train_sizes=list(train_sizes)
   train_scores_svr=list(train_scores_svr)
   test_scores_svr=list(test_scores_svr)
   #print tr_cvm
   #x1=plot_data(X)
      #session['n_data']=n_data
   session['tss']=test_scores_svr
   session['trss']=train_scores_svr
   session['ts']=train_sizes
   session['accr']=accr
   print x1
   y1=list(Y)
   y_pred=list(y_pred)
   return render_template("result4.html",x=x1,y=y1,y_pred=y_pred,trss=train_scores_svr,tss=test_scores_svr,ts=train_sizes)

@app.route('/graph_svr',methods=['GET','POST'])
def graph_svr():
  
    trn_s=session.get('trss',None)
    cv_s=session.get('tss',None)
    t_size=session.get('ts',None)

    return render_template("graph_svr.html",trss=trn_s,tss=cv_s,ts=t_size)
@app.route('/accr_svr',methods=['GET','POST'])
def accr_svr():
  accr=session.get('accr',None)
  return render_template('accr_svr.html',accr=accr)
   

@app.route('/file4',methods=['GET', 'POST'])
def index4():  
  types=[]
  header=[]
  sample=[]
  if request.method == 'POST':
     if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
     file = request.files['file']
     if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
     fi.file_path(file.filename)
     if file and fi.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           
            df = pd.read_csv(filepath)
     #header1=[]     
     with open(filepath,'r') as f:
              reader=csv.reader(f)
              header.append(reader.next())
              sample=next(reader)
     print header[0]
     '''for x in header[0]:
         pattern = re.compile(r'\s+')
         x = re.sub(pattern, '', x)
         print x
         header1.append(x)'''
     header1=np.array(header[0])
     print header1
     print df.dtypes
     session['var']=filepath
     #print (df.dtypes)
     for i in range(0, len(df.columns)):
          #print(df.dtypes[i])
            
          if(df.dtypes[i]=='int64'):
             types.append('int')
          if(df.dtypes[i]=='object'):
             types.append('varchar(100)')
          if(df.dtypes[i]=='float64'):
             types.append('real')
     print len(types)
     print sample
     c = float(request.form["c"])
     epsilon=float(request.form["epsilon"])
     gamma=float(request.form["gamma"])
     degree=int(request.form["degree"])
     kernel=str(request.form["kernel"])
     split=float(request.form["split"])
     session['c']=c #c must be graeter than 1 in case of noisy data
     session['epsilon']=epsilon
     session['gamma']=gamma
     session['degree']=degree #used in case of poly kernel
     session['kernel']=kernel
     session['split']=split
     print c,epsilon,gamma,degree,kernel,split
     return render_template("choice5.html",header=zip(header1,types,sample),file_n=file.filename)
  
  return render_template('file_render_svc.html')


@app.route('/svc',methods=['POST','GET'])
def svc():
  if request.method =='POST' or request.method=='GET':
   n_data=request.form['text']
   #n_data=request.args.get('text')
   
   print(n_data)
   
   import sqlite3
   
   con = sqlite3.connect('database.db')
   varf=session.get('var',None)
   s=session.get('split',None)
   c=session.get('c',None)
   e=session.get('epsilon',None)
   g=session.get('gamma',None)
   degree=session.get('degree',None)
   k=session.get('kernel',None)
   print (varf),c,e,g,degree,k
   data_f=varf.split('/')
   print data_f[-1]
   for x in data_f:
      dfx=x.split('.')
   sql = "DROP TABLE IF EXISTS %s" % dfx[0]
   con.execute(sql)
   query='CREATE TABLE '+ dfx[0]+'('+n_data+')'
   con.execute(query)
   data=n_data.split(',')
   var1=[]
   #print data
   for x in data:
        d=x.split(' ')
        print d
        for y in d:
           #print y 
           if(y!='int' and y!='varchar(100)' and y!='real'):
              var1.append(str(y))
   print len(var1)  
   var=tuple(var1)
   print var
   to_db=[]
   with open(varf,'rb') as fin:
      dr=csv.DictReader(fin)
      for i in dr:
        to_db.append([(i[n]) for n in var])
        #print to_db
   print len(to_db)
   q1=[]
   for i in range(0,len(var1)):
       q1.append('?')
 
   q=",".join(q1)
   print q
   quer="INSERT INTO "+dfx[0]+" "+ str(var)+" VALUES("+q+");"
   print quer
   con.executemany("INSERT INTO "+dfx[0]+" "+ str(var)+" VALUES("+(q)+");",to_db)
   con.commit()
   con.execute("PRAGMA busy_timeout = 30000")
   con.close() 
   conn = sqlite3.connect("database.db")
   df= pd.read_sql_query("select * from "+ dfx[0], conn)
   conn.close()
   conn=sqlite3.connect("database.db")
   df_n= pd.read_sql_query("select * from "+ dfx[0], conn)
   #print df[0:5]
   #print df_n[0:5]
   Y = df.iloc[:,-1].values
           
   #print df_n.dtypes
   #print len(df_n.columns)      
   df.drop(df.columns[len(df.columns)-1],axis=1,inplace=True)
   
   
   for j in range(0,len(df_n.columns)):        
              #print df_n.dtypes[j]
              #print j
              if(df_n.dtypes[j]== object):
                #print j
                df_ne=pd.get_dummies(df_n[var1[j]],drop_first=True,prefix=[str(j)],prefix_sep='_')
                #print df_ne
                df.drop(var1[j],axis=1,inplace=True)
                df=df.join(pd.DataFrame(df_ne))
                #print df.head()     
            #print  len(df.columns)
   header=[]
      
   sf=df.values.tolist()
   #print df.columns
   X = df.iloc[:,0:len(df.columns)].values
   #x1=X.flatten()
   #x=list(x)
   y_pred,cm,train_sizes,train_scores,test_scores,p,r,f,su,ll,acc=sv_c.svm(X,Y,s,k,c,g,degree)
   #print matrix[0]
   train_sizes=list(train_sizes)
   train_scores=list(train_scores)
   test_scores=list(test_scores)
   #print tr_cvm
   #x1=plot_data(X)
      #session['n_data']=n_data
   #print X
   #print X.tolist()
   session['tss']=test_scores
   session['trss']=train_scores
   session['ts']=train_sizes
   session['accr']=acc
   session['ll']=ll
   session['matrix']=cm.tolist()
   #session['report']=report
   #session['n_data']=n_data
      
   session['p']=p.tolist()
   session['r']=r.tolist()
   session['f']=f.tolist()
   session['s']=su.tolist()

   #print x1
   y1=list(Y)
   y_pred=list(y_pred)
   print train_scores
   return render_template("result5.html",trss=train_scores,tss=test_scores,ts=train_sizes)

@app.route('/accuracy_s',methods=['POST','GET'])
def accuracy_s():
    accr2=session.get('accr',None)
    #n_data=session.get('n_data',None)
    return render_template('accr_s.html',accuracy=accr2)
@app.route('/loss_s',methods=['POST','GET'])
def loss_s():
    accr1=session.get('ll',None)     
   
    #n_data=session.get('n_data',None)
    return render_template('loss_s.html',results2_m=accr1)
@app.route('/matrix_s',methods=['POST','GET'])
def matrix_s():
    lm=session.get('lm',None)
    matrix=session.get('matrix',None)     
    #accr2=session.get('results1_s',None)
    #n_data=session.get('n_data',None)
    return render_template('matrix_s.html',matrix=matrix,lm=lm)
@app.route('/report_s',methods=['POST','GET'])
def report_s():
    p=session.get('p',None)
    r=session.get('r',None)
    f=session.get('f',None)     
    s=session.get('s',None)
    #accr2=session.get('results1_s',None)
    #n_data=session.get('n_data',None)
    return render_template('report_s.html',classification=zip(p,r,f,s))
@app.route('/graph_s',methods=['POST','GET'])
def graph_s():
       
    trn_s=session.get('trss',None)
    cv_s=session.get('tss',None)
    t_size=session.get('ts',None)

    return render_template("graph_s.html",trss=trn_s,tss=cv_s,ts=t_size)



if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True, use_reloader=True)


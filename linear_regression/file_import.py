import os
from flask import Flask 
app = Flask(__name__)
UPLOAD_FOLDER = '/home/simmi'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
class file_import:

   def allowed_file(self,filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
   def file_path(self,filename):
    #print (os.path.join(app.config['UPLOAD_FOLDER'],filename))
    return (os.path.join(app.config['UPLOAD_FOLDER'],filename))

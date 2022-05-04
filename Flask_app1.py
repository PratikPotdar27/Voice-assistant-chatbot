from flask import Flask, request
import numpy as np
#import pickle
import pandas as pd
from flask import render_template
from flask import Flask, request
import numpy as np
#import pickle
import pandas as pd
import os







app=Flask(__name__)

@app.route('/', methods=["GET","POST"])
def hello():
    if request.method=="POST":
        file =request.files["file"]
        file.save(os.path.join("static",file.filename))
        return render_template('trial.html',message="success")
    return render_template('trial.html',message="upload")
    

from flask import send_file

# ... other code ....


@app.route('/file-downloads/')
def file_downloads():
	try:
		return render_template('downloads.html')
	except Exception as e:
		return str(e)



    
@app.route('/return-files/')
def return_files_tut():
	try:
		return send_file('static/tp1.csv', attachment_filename='tp1.csv')
	except Exception as e:
		return str(e)







if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)
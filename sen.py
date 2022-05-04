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
from flask import redirect, url_for
import glob






app=Flask(__name__)

@app.route('/', methods=["GET","POST"])
def hello():
    if request.method=="POST":
        file =request.files["file"]
        file.save(os.path.join("/home/ubuntu/static",file.filename))
        return redirect("p1")
        path = r'/home/ubuntu/static' # use your path
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        df_test = pd.concat(li, axis=0, ignore_index=True)
        
        
        
        
        
        
        
        
        
        #df_test= pd.read_csv('static/file1.csv')
        xx=df_test['Tweets']
        voc_size=5000
        import nltk
        nltk.download('stopwords')
        import re
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        ps = PorterStemmer()
        corpus = []
        for i in range(0, len(df_test)):
            review = re.sub('[^a-zA-Z]', ' ', df_test['Tweets'][i])
            review = review.lower()
            review = review.split()
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
        from tensorflow.keras.layers import Embedding
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.preprocessing.text import one_hot
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import Dense
    
        onehot_repr=[one_hot(words,voc_size)for words in corpus]
    
        sent_length=20
        embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
        import numpy as np
        X_final=np.array(embedded_docs)
        from tensorflow.keras.models import save_model, load_model
        filepath= './saved_model'
        model=load_model(filepath, compile = True)
        predctn= model.predict_classes(X_final)
    
        vv=df_test['Tweets'].count()
    
        gg=np.reshape(predctn,vv)
        gg1=pd.Series(gg)
        new_df=pd.DataFrame(columns=['Tweet','Sentiment'])
        new_df['Tweet']=df_test['Tweets']
        new_df['Sentiment']=gg1
    
        gf=new_df[new_df['Sentiment']==1]
        dd=gf['Sentiment'].count()
    
        gf1=new_df[new_df['Sentiment']==0]
        dd1=gf1['Sentiment'].count()
        new_df.to_csv('/home/ubuntu/final.csv')
        
        
        
        
        
        
    return render_template('trial1.html',message="success")

























@app.route('/p1')
def upload_form1():
    return render_template('p.html')

@app.route('/download1')
def download_file1():
    path="final.csv"
    return send_file(path, as_attachment=True)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)
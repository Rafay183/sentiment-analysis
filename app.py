from flask import Flask, render_template, url_for, request
import nltk
import re
from nltk.corpus import stopwords
import numpy as np
import pickle
# from nltk.corpus import stopwords
# nltk.data.path.append('/nltk_data/')

with open('nbWithtfidf', 'rb') as f:
    tfidf_model, model = pickle.load(f)
app = Flask(__name__)

NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return ' '.join(normalized_texts)

def removeStopwords(text):
    text_new=[]
    for word in text:
        if word not in stopwords.words('english'):
            text_new.append(word)
    return text_new
        
def predictSentiment(text):
    if len(text)>0:
        text = normalize_texts([text])
        # text = removeStopwords(text.split())
        # text = ' '.join(text)
        text_tfidf = tfidf_model.transform([text])
        return model.predict(text_tfidf)
    else: 
        return render_template('index.html')

@app.route('/', methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        text = request.form['content']
        Sentiment = predictSentiment(text)
        return render_template('index.html', Sentiment = Sentiment)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)
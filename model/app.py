from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import json 
import mysql.connector
import csv
import re
import nltk
import spacy
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

nlp = spacy.load('en_core_web_sm')

# Load model
model = load_model('model.h5')
embedding_dim = 64
max_length = 4 

#load data 
def load_data_func():
    filename = "sample_data/metadata.tsv"
    dictionary = []
    with open(filename, "r", newline="", encoding="utf-8") as file:
        for line in file:
            vocab = line.strip()
            dictionary.append(vocab)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dictionary)
    return tokenizer

#Pre processing
def remove_special_characters(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text


def convert_to_lowercase(sentence):
    sentence = sentence.lower().strip()
    return sentence

def expand_sentence(sentence):
  sentence = contractions.fix(sentence)
  return sentence

def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))

    tokens = sentence.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(filtered_tokens)
    return text

def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    lemmatized_words = ' '.join([token.lemma_ if token.tag_ != 'NNS' else lemmatizer.lemmatize(token.text) for token in doc]) 
    return lemmatized_words

def pre_processing(sentence):
  expanded_sentence = expand_sentence(sentence)
  lemmatized_sentence = lemmatize_sentence(expanded_sentence)
  lowercase_sentence = convert_to_lowercase(lemmatized_sentence)
  removed_stopwords = remove_stopwords(lowercase_sentence)
  removed_special_characters = remove_special_characters(removed_stopwords)
  return removed_special_characters

#Tokenizer 
def tokenizer_func(text): 
    tokenizer = load_data_func()
    text_seq = tokenizer.texts_to_sequences([text])
    return text_seq

def get_department_id(): 
    # database = mysql.connector.connect(
    #     host='mysql-springboot-container',
    #     user='root',
    #     password='12345678',
    #     database='hospitalCareDB'
    # )
    database = mysql.connector.connect(
        host='localhost',
        user='root',
        password='12345678',
        database='hospital_care'
    )
    cursor = database.cursor()
    cursor.execute("SELECT id FROM departments")
    departments = cursor.fetchall()
    department_id = []
    for department in departments: 
        department_id.append(department[0])
    return department_id

#Khoi tao Flask Server Backend 
app = Flask(__name__)

#Apply Flask CORS 
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/adds', methods=['POST'])
@cross_origin(origin='*')
def add(): 
    s = request.form.get('text')
    return s

@app.route('/api/v1/python/search_doctor', methods=['POST'])
def predict():
    data = request.json
    searchText = data.get('searchText')
    searchTextPre = pre_processing(searchText)
    text_seq = tokenizer_func(searchTextPre)
    padded_test_seq = pad_sequences(text_seq, maxlen=max_length, truncating='post', padding='post')
    prediction = model.predict(padded_test_seq)
    
    department_predict = []
    
    department_id = get_department_id()
    for item in prediction[0]: 
        if item > 0.5: 
            department_predict.append(department_id[np.where(prediction == item)[1][0]])
    return {"departmentId": department_predict}

@app.route('/api/v1/test')
def test():
    return "success"
#Start Backend 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')
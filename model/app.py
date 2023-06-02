from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import json 
import mysql.connector

# Load model
model = load_model('model.h5')
vocab_size = 1000
embedding_dim = 64
max_length = 140 

#load data 
def load_data_func():
    with open('sample_data/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

#sentences vs labels 
def assign_sentences_labels(data): 
    sentences = []
    labels = []

    for item in data: 
        sentences.append(item["text"])
        labels.append(item["label"])
    
    labels = np.array(labels)
    return [sentences, labels]

#Tokenizer 
def tokenizer_func(text): 
    data = load_data_func()
    train_data = assign_sentences_labels(data)
    train_sentences = train_data[0]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_sentences)
    text_seq = tokenizer.texts_to_sequences([text])
    return text_seq

def get_department_id(): 
    database = mysql.connector.connect(
        host='mysql-springboot-container',
        user='root',
        password='12345678',
        database='hospitalCareDB'
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
    text_seq = tokenizer_func(searchText)
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
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port='6868')
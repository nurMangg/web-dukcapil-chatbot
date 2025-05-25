from flask import Flask, render_template, request, jsonify
import pickle
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import os
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/chatbot_disdukcapil'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the trained model and data
try:
    # Load all data from the single pickle file
    chatbot_data = pickle.load(open('chatbot_data.pickle', 'rb'))
    model = chatbot_data['model']
    classes = chatbot_data['classes']
    intents = chatbot_data['intents']
    model_loaded = True
    print("Model loaded successfully!")
except (FileNotFoundError, IOError) as e:
    print(f"Error loading model: {e}")
    print("Using fallback responses.")
    # Load just the intents for fallback
    try:
        intents = json.loads(open('intents.json').read())
    except (FileNotFoundError, IOError):
        intents = {"intents": []}
    model_loaded = False
    
class ChatLog(db.Model):
       id = db.Column(db.Integer, primary_key=True)
       user_message = db.Column(db.String(500))
       bot_response = db.Column(db.String(500))
       intent = db.Column(db.String(100))
       probability = db.Column(db.Float)
       timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
       def __repr__(self):
           return f'<ChatLog {self.user_message}>'
       
def create_tables():
    with app.app_context():
        db.create_all() 

def predict_class(sentence):
    if not model_loaded:
        # Fallback to keyword matching if model is not loaded
        return keyword_matching(sentence)
    
    # Use the trained model to predict the intent
    try:
        # The model is a pipeline that includes vectorization and classification
        intent_idx = model.predict([sentence])[0]
        return [{'intent': intent_idx, 'probability': '0.9'}]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return [{'intent': 'fallback', 'probability': '1.0'}]

def keyword_matching(sentence):
    sentence = sentence.lower()
    
    # Define keywords for each intent
    keywords = {
        'sapaan': ['halo', 'hai', 'selamat', 'pagi', 'siang', 'sore', 'malam'],
        'ucapan_sampai': ['sampai', 'jumpa', 'selamat', 'tinggal', 'terima kasih'],
        'syarat': ['syarat', 'dokumen', 'apa', 'harus', 'saya'],
        'proses': ['proses', 'bagaimana', 'langkah', 'step', 'cara'],
        'biaya': ['biaya', 'harga', 'berapa', 'bayar', 'uang'],
        'lokasi': ['kantor', 'alamat', 'dimana', 'lokasi', 'tempat'],
        'online': ['online', 'website', 'internet', 'digital'],
        'waktu': ['waktu', 'berapa', 'lama', 'hari', 'minggu'],
        'koreksi': ['koreksi', 'ganti', 'ubah', 'perbaiki', 'perbarui'],
        'duplikat': ['duplikat', 'salinan', 'ganda', 'kehilangan', 'lagi']
    }
    
    # Check for keyword matches
    for intent, words_list in keywords.items():
        if any(word in sentence for word in words_list):
            return [{'intent': intent, 'probability': '0.8'}]
    
    # Default to fallback
    return [{'intent': 'fallback', 'probability': '1.0'}]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        # If no matching intent is found
        fallback_responses = [
            "Maaf, saya tidak mengerti. Bisakah Anda mengulang pertanyaan Anda tentang sertifikat kelahiran?",
            "Saya tidak memiliki informasi tentang itu. Bisakah Anda menanyakan sesuatu yang terkait dengan aplikasi sertifikat kelahiran?"
        ]
        result = random.choice(fallback_responses)
    
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    message = request.json['message']
    ints = predict_class(message)
    response = get_response(ints, intents)
    # Simpan ke database
    chat_log = ChatLog(
        user_message=message,
        bot_response=response,
        intent=ints[0]['intent'] if ints else 'unknown',
        probability=float(ints[0]['probability']) if ints else 0.0
    )
    db.session.add(chat_log)
    db.session.commit()
    return jsonify({'response': response})

if __name__ == '__main__':
    create_tables()
    port = int(os.environ.get('PORT', 5000))  # default ke 5000 jika PORT tidak disetel
    app.run(host='0.0.0.0', port=port)


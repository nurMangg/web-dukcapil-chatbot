import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Initialize lists
documents = []
classes = []
ignore_letters = ['?', '!', '.', ',']

# Load the intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Process the intents data
X = []  # patterns
y = []  # tags

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Add to documents
        X.append(pattern)
        y.append(intent['tag'])
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Sort classes
classes = sorted(list(set(classes)))

# Create a pipeline with CountVectorizer and MultinomialNB
model = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=nltk.word_tokenize, lowercase=True)),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X, y)

# Save the model and classes
pickle.dump(model, open('chatbot_model.pickle', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create a dictionary to store all the data
chatbot_data = {
    'classes': classes,
    'model': model,
    'intents': intents
}

# Save all data in a single pickle file
pickle.dump(chatbot_data, open('chatbot_data.pickle', 'wb'))

print("Model trained and saved successfully!")

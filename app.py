import streamlit as st
import numpy as np
import re  # Add this import statement for the 're' module
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('twitter_sentiment.h5')

# Load the tokenizer
tokenizer = Tokenizer(num_words=10000, split=' ')
# Assuming you saved your tokenizer during training, load it here
# tokenizer = load_tokenizer()  # Uncomment and replace with the actual loading code

# Function to preprocess text input
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-z0-9\s]', '', text)
    return text

# Function to predict sentiment
def predict_sentiment(text, tokenizer, max_length):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    # Padding needs to be done with the same tokenizer used for training
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(np.array(padded_sequence))
    return prediction[0][0]

# Streamlit App
st.title("Twitter Sentiment Analysis")

# Text input for the user to enter a tweet
tweet_input = st.text_area("Enter a tweet:")

# Button to trigger sentiment analysis
if st.button("Predict Sentiment"):
    if tweet_input:
        # Tokenize the input text using the loaded tokenizer
        sequence = tokenizer.texts_to_sequences([tweet_input])
        # Get the maximum sequence length expected by the model
        max_length = model.input_shape[1]
        # Padding needs to be done with the same tokenizer used for training
        X = pad_sequences(sequence, maxlen=max_length)
        
        sentiment_score = predict_sentiment(tweet_input, tokenizer, max_length)

        # Display sentiment
        sentiment_class = "Positive" if sentiment_score > 0.5 else "Negative"
        st.success(f"Sentiment: {sentiment_class}")
    else:
        st.warning("Please enter a tweet for sentiment analysis.")
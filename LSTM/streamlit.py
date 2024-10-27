import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model("model.h5")

# Load the tokenizer
with open("token.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, input_text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = np.argmax(model.predict(token_list), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

# Streamlit UI
st.title("Next Word Prediction App")
st.write("Enter some words, and the model will predict the next word.")

# User input
input_text = st.text_input("Enter your text here:")

# Display predicted word
if input_text:
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted Next Word: **{next_word}**")

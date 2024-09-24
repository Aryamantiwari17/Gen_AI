import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding,SimpleRNN
from tensorflow.keras.models import load_model
import streamlit as st


##Mapping of words index back to words
word_index=imdb.get_word_index()
#word_index

#reversing it for better mapping
reverse_word={value:key for key,value in word_index.items()}
#reverse_word
model=load_model('simple_rnn.h5')
#model.summary()
##we need a helper function
#function to decode reviews

def decode_review(encoded_review):
  return ' '.join([reverse_word.get(i-3,'?') for i in encoded_review])



def preprocessing_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word,2)+3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review
  

##prediciton function
def predict_as(review):
   pre_process_input= preprocessing_text(review)

   prediction=model.predict(pre_process_input)

   sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

   return sentiment,prediction[0][0]


# Streamlit App
def main():
    st.title("IMDB Movie Review Sentiment Analysis")
    
    # Input text from the user
    review_text = st.text_area("Enter your movie review:")
    
    if st.button('Analyze Sentiment'):
        sentiment, score = predict_as(review_text)
        
        # Display results
        st.write(f"**Review:** {review_text}")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {score}")
        
        # Optionally decode the review
       # st.write("**Decoded Review (Tokenized):**")
      #  st.write(decode_review(preprocessing_text(review_text)[0]))

if __name__ == '__main__':
    main()

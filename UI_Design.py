import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model  
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved files
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    
    return tokenizer, model, max_len 
tokenizer, model, max_len = load_resources()

# Prediction function

def predict_next_word(input_text):
    # Step 1: Tokenize
    sequence = tokenizer.texts_to_sequences([input_text])
    
    # 🚨 Handle empty or invalid input
    if not sequence or len(sequence[0]) == 0:
        return "No prediction (unknown words)"
    
    # Step 2: Pad properly
    sequence = pad_sequences(sequence, maxlen=max_len-1, padding='pre')
    
    # Step 3: Predict
    preds = model.predict(sequence, verbose=0)
    confidence = np.max(preds)
    predicted_index = np.argmax(preds[0])
    
    # Step 4: Fast lookup (better than loop)
    return tokenizer.index_word.get(predicted_index, "Word not found")

# Streamlit UI


st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="auto"
)
st.title("🔮 Next Word Prediction")
st.write("Enter a sequence of words, and the model will predict the next word in the sequence.")
user_input = st.text_input("Enter your text here:", placeholder="Type a sequence of words...")
if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict the next word.")
    else:
        next_word = predict_next_word(user_input)
        st.success(f"The predicted next word is: **{next_word}**")
        
## Footer _____________________
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")


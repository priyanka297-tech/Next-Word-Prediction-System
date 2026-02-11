import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences


# Load saved files
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len   # ✅ order fixed


# Correct order
model, tokenizer, max_len = load_resources()


# Prediction function
def predict_next_word(input_text):

    # Tokenize
    sequence = tokenizer.texts_to_sequences([input_text])[0]   # ✅ fixed name

    # Pad (must be 2D)
    sequence = pad_sequences(
        [sequence],              # ✅ wrapped in list
        maxlen=max_len - 1,
        padding="pre"
    )

    # Predict
    preds = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(preds)

    # Index → Word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return ""


# ============================
# Streamlit UI
# ============================

st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Next Word Prediction")

st.write(
    "Enter a sequence of words, and the model will predict the next word."
)

user_input = st.text_input(
    "Enter your text here:",
    placeholder="Type a sequence of words..."
)

if st.button("Predict Next Word"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        next_word = predict_next_word(user_input)

        st.success(f"Predicted next word: **{next_word}**")


# Footer
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")

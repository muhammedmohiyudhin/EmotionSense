import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "saved_model")

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


labels = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']

st.set_page_config(page_title="EmotionSense", layout="centered")

st.title("EmotionSense – Emotion Detection System")
st.write("Enter a sentence and the system will detect the emotion.")

text = st.text_area("Enter your text here:")

if st.button("Analyze Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs).item()

        st.success(f"Predicted Emotion: {labels[pred_id]}")
        st.info(f"Confidence: {probs[0][pred_id].item()*100:.2f}%")


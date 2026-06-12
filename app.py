import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "spam_classifier_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

st.title("📧 Email Spam Detection using BERT")

email_text = st.text_area(
    "Paste Email Content Here",
    height=200
)

if st.button("Predict"):

    if email_text.strip() == "":
        st.warning("Please enter email text.")
    else:

        inputs = tokenizer(
            email_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

        if prediction == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam")
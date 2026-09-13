import streamlit as st
from transformers import pipeline

st.title("AI Sentiment Analyzer")

st.write("Enter a sentence and check whether it is positive or negative.")

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

text = st.text_area("Enter your text")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text")

    else:
        result = sentiment(text)[0]

        label = result["label"]
        score = result["score"]

        if label == "POSITIVE":
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

        st.write("Sentiment:", label)
        st.write("Confidence:", f"{score:.2%}")
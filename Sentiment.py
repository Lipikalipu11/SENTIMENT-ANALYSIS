{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pickle\
\
# Function to load the model and vectorizer\
def load_model_and_vectorizer():\
    with open("transformer.pkl", "rb") as f:\
        vectorizer = pickle.load(f)\
    with open("model.pkl", "rb") as f:\
        ensemble_model = pickle.load(f)\
    return vectorizer, ensemble_model\
\
# Function to get sentiment\
def get_sentiment(reviewbody):\
    vectorizer, ensemble_model = load_model_and_vectorizer()\
    x = vectorizer.transform([reviewbody])\
    y = ensemble_model.predict(x)\
    return y[0]  # Return the first (and only) prediction\
\
# Streamlit app\
def main():\
    st.title("Sentiment Analysis App")\
    st.write("Enter a review to predict its sentiment")\
\
    review = st.text_area("Review", "")\
\
    if st.button("Predict Sentiment"):\
        if review.strip() == "":\
            st.write("Please enter a review.")\
        else:\
            sentiment = get_sentiment(review)\
            st.write(f"This is a \{sentiment\} sentiment!")\
\
if __name__ == "__main__":\
    main()\
}
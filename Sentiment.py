import streamlit as st
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the trained TF-IDF vectorizer and SVM model
vector = pickle.load(open('./model/vector.pkl', 'rb'))
model = pickle.load(open('./model/svm_model.pkl', 'rb'))

# Define a function for prediction
def predict_sentiment(review):
    # Transform the input review using the TF-IDF vectorizer
    transformed_review = vector.transform([review])
    # Convert sparse matrix to dense
    dense_review = transformed_review.toarray()
    # Predict the sentiment
    prediction = model.predict(dense_review)
    # Map the prediction to the corresponding sentiment
    sentiment_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    return sentiment_map[prediction[0]]

# Streamlit app title and description
st.title('Amazon Product Review Sentiment Analysis')
st.write('Enter a product review to predict its sentiment (Positive, Neutral, or Negative).')

# Text input for the user to enter a review
user_input = st.text_area('Enter a product review:')

# Predict sentiment when button is pressed
if st.button('Predict Sentiment'):
    if user_input.strip() != '':
        sentiment = predict_sentiment(user_input)
        st.write(f'The sentiment is: *{sentiment}*')
    else:
        st.write("Please enter a review to predict.")

# Optional: Add information about the model
st.sidebar.title("About")
st.sidebar.write("""
This app uses a Support Vector Machine (SVM) model trained on Amazon product reviews to classify the sentiment of new reviews as positive, neutral, or negative. The text input is processed using TF-IDF vectorization before being fed into the model.
""")

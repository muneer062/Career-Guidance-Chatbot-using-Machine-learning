
import streamlit as st
import joblib
import re
import string
import pandas as pd

# Load the trained model and vectorizer
# These models are loaded once when the app starts to avoid reloading on each interaction.
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load the dataset to retrieve answers
# The dataset is loaded to fetch the 'answer' corresponding to the predicted 'role'.
df = pd.read_csv("career_guidance_dataset.csv")

# Text preprocessing function (must be the same as in train_model.py)
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r'\\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Function to get career suggestion
def get_career_suggestion(user_input):
    # Preprocess the user's input
    cleaned_input = preprocess_text(user_input)
    # Transform the cleaned input using the loaded TF-IDF vectorizer
    input_tfidf = vectorizer.transform([cleaned_input])
    # Predict the career role using the loaded model
    predicted_role = model.predict(input_tfidf)[0]

    # Filter the DataFrame to get answers for the predicted role
    # This ensures that the chatbot provides relevant information for the identified career.
    role_answers = df[df["role"] == predicted_role]["answer"].tolist()

    if role_answers:
        # Return a combination of the predicted role and a sample answer
        return f"Based on your interest, you might be interested in a career as a **{predicted_role}**.\n\nHere's some information: {role_answers[0]}"
    else:
        return "I'm sorry, I couldn't find information for that career role. Please try rephrasing your question."

# Streamlit UI
st.title("Career Guidance Chatbot")

# Text input for user question
user_question = st.text_input("Ask me about career paths (e.g., 'What does a Data Scientist do?', 'Tell me about software engineering'):")

# Button to get suggestion
if st.button("Get Career Suggestion"):
    if user_question:
        # Get and display the career suggestion
        suggestion = get_career_suggestion(user_question)
        st.write(suggestion)
    else:
        st.write("Please enter a question to get a career suggestion.")



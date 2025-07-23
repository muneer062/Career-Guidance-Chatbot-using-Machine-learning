
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import re
import string

# Load the dataset
def load_data(filepath):
    # Load the career guidance dataset from the specified CSV file.
    df = pd.read_csv(filepath)
    return df

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Train the model
def train_model(df):
    # Apply preprocessing to the 'question' column
    df['cleaned_question'] = df['question'].apply(preprocess_text)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_question'], df['role'], test_size=0.2, random_state=42)

    # Initialize TF-IDF Vectorizer
    # Convert text data into TF-IDF features.
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Logistic Regression model
    # Choose Logistic Regression for its simplicity and effectiveness in text classification.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model F1-score: {f1:.2f}")

    return model, vectorizer

# Main execution
if __name__ == "__main__":
    # Define the path to the dataset
    filepath = 'career_guidance_dataset.csv'
    # Load the dataset
    df = load_data(filepath)
    # Train the model and get the trained model and vectorizer
    model, vectorizer = train_model(df)

    # Save the trained model and vectorizer
    # Use joblib to save the model and vectorizer for later use in the Streamlit app.
    joblib.dump(model, 'intent_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Model and vectorizer saved successfully.")



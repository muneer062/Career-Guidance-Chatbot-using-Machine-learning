# Career Guidance Chatbot using Machine Learning

## Project Overview

This project implements a Career Guidance Chatbot that uses Machine Learning to understand user interests through natural language input and suggests relevant career fields. The chatbot is built using a trained intent classification model and features a user-friendly Streamlit web interface.

## Features

- **Natural Language Processing**: Understands user questions about career paths
- **Machine Learning Classification**: Uses TF-IDF vectorization and Logistic Regression for career role prediction
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Comprehensive Career Database**: Covers 54+ different career roles with detailed information
- **Real-time Predictions**: Instant career suggestions based on user input

## Dataset Information

The project uses a Career Guidance Dataset containing:
- **1,620+ rows** of career-related data
- **54 different career roles** (Data Scientist, Product Manager, Software Engineer, etc.)
- **Three main columns**:
  - `Role`: Career role name
  - `Question`: User questions about that role
  - `Answer`: Informative responses for career guidance

## Project Structure

```
career_chatbot_project/
├── app.py                          # Streamlit frontend application
├── train_model.py                  # Model training script
├── career_guidance_dataset.csv     # Training dataset
├── intent_model.pkl               # Saved trained model
├── vectorizer.pkl                 # Saved TF-IDF vectorizer
└── README.md                      # Project documentation
```

## Technical Implementation

### 1. Text Preprocessing
- **Lowercase conversion**: Standardizes text input
- **Punctuation removal**: Cleans text for better feature extraction
- **Number removal**: Eliminates numeric characters
- **Whitespace normalization**: Removes extra spaces

### 2. Feature Engineering
- **TF-IDF Vectorization**: Converts text into numerical features
- **Scikit-learn TfidfVectorizer**: Handles term frequency-inverse document frequency calculation

### 3. Machine Learning Model
- **Algorithm**: Logistic Regression
- **Performance**: 100% accuracy and F1-score on test set
- **Training/Testing Split**: 80/20 ratio
- **Model Persistence**: Saved using joblib for deployment

### 4. Web Interface
- **Framework**: Streamlit
- **Features**: Text input, prediction button, formatted output
- **Real-time Processing**: Instant career suggestions

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Packages
```bash
pip install pandas scikit-learn joblib streamlit
```

### Installation Steps

1. **Clone or download the project files**
2. **Navigate to the project directory**:
   ```bash
   cd career_chatbot_project
   ```
3. **Install dependencies**:
   ```bash
   pip install pandas scikit-learn joblib streamlit
   ```

## Usage Instructions

### Training the Model

1. **Run the training script**:
   ```bash
   python train_model.py
   ```
   
   This will:
   - Load and preprocess the dataset
   - Train the Logistic Regression model
   - Evaluate model performance
   - Save the trained model and vectorizer

### Running the Chatbot

1. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**:
   - Open your browser and go to `http://localhost:8501`
   - The Career Guidance Chatbot interface will load

3. **Using the chatbot**:
   - Enter your career-related question in the text input
   - Click "Get Career Suggestion"
   - View the predicted career role and detailed information

### Example Queries

- "What does a Data Scientist do?"
- "Tell me about software engineering"
- "What skills do I need for product management?"
- "How do I become a UX designer?"

## Model Performance

The trained model achieves excellent performance metrics:
- **Accuracy**: 100%
- **F1-Score**: 100%

These high scores indicate that the model can perfectly classify career-related questions in the test dataset, making it highly reliable for career guidance applications.

## Code Documentation

### train_model.py

```python
# Main functions and their purposes:

def load_data(filepath):
    # Loads the career guidance dataset from CSV file
    
def preprocess_text(text):
    # Cleans and standardizes text input for model training
    
def train_model(df):
    # Trains the Logistic Regression model and evaluates performance
```

### app.py

```python
# Main functions and their purposes:

def preprocess_text(text):
    # Same preprocessing function used in training (consistency is crucial)
    
def get_career_suggestion(user_input):
    # Processes user input and returns career predictions with explanations
```

## Technical Details

### Text Preprocessing Pipeline
1. **Lowercase Conversion**: `text.lower()`
2. **Punctuation Removal**: `str.maketrans("", "", string.punctuation)`
3. **Number Removal**: `re.sub(r'\\d+', '', text)`
4. **Whitespace Normalization**: `re.sub(r'\\s+', ' ', text).strip()`

### Machine Learning Pipeline
1. **Data Splitting**: 80% training, 20% testing
2. **Feature Extraction**: TF-IDF vectorization
3. **Model Training**: Logistic Regression with max_iter=1000
4. **Model Evaluation**: Accuracy and F1-score metrics
5. **Model Persistence**: joblib serialization

### Streamlit Interface Components
1. **Title**: "Career Guidance Chatbot"
2. **Text Input**: User question input field
3. **Button**: "Get Career Suggestion" trigger
4. **Output Display**: Formatted career suggestions

## Future Enhancements

### Potential Improvements
1. **Enhanced NLP**: Implement more sophisticated text preprocessing
2. **Model Diversity**: Experiment with other algorithms (SVM, Random Forest, Neural Networks)
3. **User Interface**: Add more interactive features and better styling
4. **Database Integration**: Connect to live career databases for updated information
5. **Personalization**: Implement user profiles and recommendation history
6. **Multi-language Support**: Extend to support multiple languages

### Scalability Considerations
1. **Model Retraining**: Implement automated retraining with new data
2. **Performance Optimization**: Add caching for faster response times
3. **Deployment**: Containerize the application for cloud deployment
4. **Monitoring**: Add logging and performance monitoring

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   - Ensure all required packages are installed
   - Check Python version compatibility

2. **Model Loading Errors**:
   - Verify that `intent_model.pkl` and `vectorizer.pkl` exist
   - Run `train_model.py` first to generate model files

3. **Streamlit Port Issues**:
   - Use different port: `streamlit run app.py --server.port 8502`
   - Check if port 8501 is already in use

4. **Dataset Loading Errors**:
   - Ensure `career_guidance_dataset.csv` is in the correct directory
   - Check file permissions and format

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Test thoroughly
5. Submit a pull request

## License

This project is created for educational purposes as part of a Machine Learning internship program.

## Author

**Manus AI** - Machine Learning Internship Project

## Acknowledgments

- Dataset provided as part of the Summer Internship 2025 Machine Learning program
- Streamlit framework for the web interface
- Scikit-learn library for machine learning implementation


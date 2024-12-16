import pickle
from feature_engineering import vectorize_text
from preprocess import clean_text

# Load the necessary artifacts
def load_artifacts(model_path, vectorizer_path):
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    return model, vectorizer

# Predict function for deployment
def predict_text(model, vectorizer, input_text):
    # Preprocess and vectorize the input text
    cleaned_text = clean_text(input_text)
    tfidf_features = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_features)
    return prediction

if __name__ == "__main__":
    model_path = "model_artifacts/model_logreg.pkl"
    vectorizer_path = "model_artifacts/tfidf_vectorizer.pkl"
    
    # Load artifacts
    model, vectorizer = load_artifacts(model_path, vectorizer_path)
    
    # Input text for prediction
    input_text = "I am so happy with the service!"
    prediction = predict_text(model, vectorizer, input_text)
    
    print(f"Predicted Intensity: {prediction[0]}")

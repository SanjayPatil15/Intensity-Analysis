import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from feature_engineering import vectorize_text
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split

# Load data and preprocess
def load_data(data_folder):
    angriness_df = pd.read_csv(f"{data_folder}/angriness.csv")
    happiness_df = pd.read_csv(f"{data_folder}/happiness.csv")
    sadness_df = pd.read_csv(f"{data_folder}/sadness.csv")
    combined_df = pd.concat([angriness_df, happiness_df, sadness_df], ignore_index=True)
    return preprocess_data(combined_df)

# Load models and vectorizer
def load_artifacts(model_path, vectorizer_path):
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    return model, vectorizer

# Evaluate the model
def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    data_folder = "data"
    model_artifacts_folder = "model_artifacts"
    
    # Load data
    df = load_data(data_folder)
    X = df['cleaned_content']
    y = df['intensity']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Load vectorizer and transform data
    _, vectorizer = load_artifacts(None, f"{model_artifacts_folder}/tfidf_vectorizer.pkl")
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Evaluate each model
    for model_name in ['logreg', 'rf', 'svm', 'nn']:
        print(f"\nEvaluating {model_name.upper()} model:")
        model, _ = load_artifacts(f"{model_artifacts_folder}/model_{model_name}.pkl", None)
        evaluate_model(model, X_test_tfidf, y_test)

from evaluate import evaluate_model, load_artifacts
from deploy import predict_text
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engineering import vectorize_text
from preprocess import preprocess_data

def main():
    # Step 1: Load Data
    data_folder = "data"
    model_artifacts_folder = "model_artifacts"
    
    print("Loading and preprocessing data...")
    angriness_df = pd.read_csv(f"{data_folder}/angriness.csv")
    happiness_df = pd.read_csv(f"{data_folder}/happiness.csv")
    sadness_df = pd.read_csv(f"{data_folder}/sadness.csv")
    combined_df = pd.concat([angriness_df, happiness_df, sadness_df], ignore_index=True)
    combined_df = preprocess_data(combined_df)
    
    X = combined_df['cleaned_content']
    y = combined_df['intensity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Step 2: Load Artifacts
    _, vectorizer = load_artifacts(None, f"{model_artifacts_folder}/tfidf_vectorizer.pkl")
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("\nEvaluating Models...")
    for model_name in ['logreg', 'rf', 'svm', 'nn']:
        print(f"\nEvaluating {model_name.upper()} model:")
        model, _ = load_artifacts(f"{model_artifacts_folder}/model_{model_name}.pkl", None)
        evaluate_model(model, X_test_tfidf, y_test)
    
    # Step 3: Predict on New Text
    input_text = "The service was horrible and I am very angry!"
    model, _ = load_artifacts(f"{model_artifacts_folder}/model_logreg.pkl", None)
    prediction = predict_text(model, vectorizer, input_text)
    
    print(f"\nPrediction for '{input_text}': {prediction[0]}")

if __name__ == "__main__":
    main()

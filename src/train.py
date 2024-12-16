import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data
from feature_engineering import vectorize_text
from models import logistic_regression, random_forest, svm, neural_network
import pandas as pd

# Load and preprocess data
def load_data(data_folder):
    angriness_df = pd.read_csv(f"{data_folder}/angriness.csv")
    happiness_df = pd.read_csv(f"{data_folder}/happiness.csv")
    sadness_df = pd.read_csv(f"{data_folder}/sadness.csv")
    combined_df = pd.concat([angriness_df, happiness_df, sadness_df], ignore_index=True)
    return preprocess_data(combined_df)

# Main training function
def train_models(data_folder, save_path):
    # Load data
    df = load_data(data_folder)
    X = df['cleaned_content']
    y = df['intensity']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Vectorize text
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    
    # Logistic Regression
    logreg = logistic_regression()
    logreg.fit(X_train_tfidf, y_train)
    pickle.dump(logreg, open(f"{save_path}/model_logreg.pkl", "wb"))
    
    # Random Forest
    rf = random_forest()
    rf.fit(X_train_tfidf, y_train)
    pickle.dump(rf, open(f"{save_path}/model_rf.pkl", "wb"))
    
    # SVM
    svm_model = svm()
    svm_model.fit(X_train_tfidf, y_train)
    pickle.dump(svm_model, open(f"{save_path}/model_svm.pkl", "wb"))
    
    # Neural Network
    nn_model = neural_network(X_train_tfidf.shape[1])
    nn_model.fit(X_train_tfidf.toarray(), y_train, epochs=10, batch_size=32, verbose=1)
    nn_model.save(f"{save_path}/model_nn.pkl")
    
    # Save vectorizer
    pickle.dump(vectorizer, open(f"{save_path}/tfidf_vectorizer.pkl", "wb"))
    print("Models trained and saved successfully!")

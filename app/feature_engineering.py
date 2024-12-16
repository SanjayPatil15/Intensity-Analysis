from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
def initialize_vectorizer():
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")

# Fit and transform data
def vectorize_text(vectorizer, text_data):
    return vectorizer.fit_transform(text_data)

# Transform new text data
def transform_text(vectorizer, new_text):
    return vectorizer.transform(new_text)

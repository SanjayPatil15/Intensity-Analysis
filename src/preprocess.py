import pandas as pd

def clean_text(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')  # Remove emojis and special characters
    text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Keep alphanumeric and spaces
    return text.lower().strip()

def preprocess_data(df):
    df['cleaned_content'] = df['content'].apply(clean_text)
    return df

from flask import Flask, render_template, request
import pickle
from preprocess import preprocess_data
from feature_engineering import vectorize_text

app = Flask(__name__)

# Load the model and vectorizer once
model = pickle.load(open('app/model/best_mlp_model.pkl', 'rb'))
vectorizer = pickle.load(open('app/model/tfidf_vectorizer.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Preprocess and vectorize the input text
        preprocessed_text = preprocess_data(user_input)
        vectorized_input = vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = model.predict(vectorized_input)

        # Convert prediction to readable format
        intensity = {0: "Sadness", 1: "Happiness", 2: "Angriness"}
        result = intensity.get(prediction[0], "Unknown")

        return render_template('index.html', user_input=user_input, result=result)
    
    # Handle GET request - return the initial page
    return render_template('index.html', user_input=None, result=None)


if __name__ == "__main__":
    app.run(debug=True)

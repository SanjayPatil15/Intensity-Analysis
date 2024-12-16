from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Logistic Regression
def logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=42)

# Random Forest
def random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42)

# Support Vector Machine
def svm():
    return SVC(kernel='linear', probability=True, random_state=42)

# Neural Network
def neural_network(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

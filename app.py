import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st

# Load and preprocess data
def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    data.columns = ["RL", "AG", "CH", "DC", "CE", "SC", "PT", "PE", "LP"]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

    # Encode categorical data
    encoder = LabelEncoder()
    data['LP'] = encoder.fit_transform(data['LP'])
    
    return data, encoder

# Train and save the model
def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy}")

    # Save model to file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model

# Streamlit app
def main():
    st.title("Predictive Analysis with Decision Tree")

    # Load the saved model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Define inputs
    RL = st.number_input("RL", value=0.0, step=0.1)
    AG = st.number_input("AG", value=0.0, step=0.1)
    CH = st.number_input("CH", value=0.0, step=0.1)
    DC = st.number_input("DC", value=0.0, step=0.1)
    CE = st.number_input("CE", value=0.0, step=0.1)
    SC = st.number_input("SC", value=0.0, step=0.1)
    PT = st.number_input("PT", value=0.0, step=0.1)
    PE = st.number_input("PE", value=0.0, step=0.1)

    if st.button("Predict"):
        input_data = np.array([[RL, AG, CH, DC, CE, SC, PT, PE]])
        prediction = model.predict(input_data)
        st.write("Prediction (Encoded):", prediction[0])

if __name__ == "__main__":
    main()

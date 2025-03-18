import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    df = pd.DataFrame(data=X, columns=iris.feature_names)
    df['target'] = y
    return df

# Exploratory Data Analysis
def explore_data(df):
    print("Data Shape: ", df.shape)
    print("First 5 Rows of Data:\n", df.head())
    print("Data Description:\n", df.describe())
    print("Data Info:\n", df.info())
    
    # Visualization
    sns.pairplot(df, hue='target')
    plt.show()

# Preprocessing the Data
def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Building the Model
def build_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

# Training the Model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Evaluating the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

# Saving the Model
def save_model(model, filename):
    joblib.dump(model, filename)

# Main Function
def main():
    df = load_data()
    explore_data(df)
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = build_model()
    
    trained_model = train_model(model, X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)
    
    save_model(trained_model, 'random_forest_model.pkl')
    print("Model saved successfully!")

# Running the Main Function
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Custom styling
st.markdown("""
<style>
.main { background-color: #f5f5f5; }
.sidebar .sidebar-content { background-color: #e0f7fa; }
</style>
""", unsafe_allow_html=True)

# Load dataset and model
@st.cache_data
def load_data():
    file_path = 'data/Titanic-Dataset.csv'
    if not os.path.exists(file_path):
        st.error(f"Dataset file not found at {file_path}")
        return None
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def load_model():
    try:
        with open('notebooks/model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

df = load_data()
model = load_model()
if df is None or model is None:
    st.stop()

# Preprocessing function
def preprocess_data(df, for_prediction=False):
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'] + (['Survived'] if not for_prediction else []), axis=1, errors='ignore')
    X['Age'] = X['Age'].fillna(X['Age'].median(), inplace=True)
    X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    expected_cols = ['Pclass','Age','SibSp','Parch','Fare','FamilySize','Sex_male','Embarked_Q','Embarked_S']
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]
    return X

# App title and description
st.title("Titanic Survival Prediction App")
st.write("""
This app predicts whether a passenger survived the Titanic disaster using a Random Forest model.
Explore the dataset, visualize key insights, and make predictions!
""")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Visualizations", "Prediction", "Model Performance"])

if page == "Data Exploration":
    st.header("Data Exploration")
    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", list(df.columns))
    st.write("Data Types:\n", df.dtypes)
    st.subheader("Sample Data")
    st.dataframe(df.head())
    st.subheader("Filter Data")
    pclass_filter = st.multiselect("Select Passenger Class", options=df['Pclass'].unique(), default=df['Pclass'].unique())
    filtered_df = df[df['Pclass'].isin(pclass_filter)]
    st.dataframe(filtered_df)

if page == "Visualizations":
    st.header("Visualizations")
    st.subheader("Survival by Passenger Class")
    fig1 = px.histogram(df, x='Pclass', color='Survived', barmode='group')
    st.plotly_chart(fig1)

    st.subheader("Survival by Sex")
    fig2 = px.histogram(df, x='Sex', color='Survived', barmode='group')
    st.plotly_chart(fig2)

    st.subheader("Age Distribution")
    fig3 = px.histogram(df, x='Age', nbins=30)
    st.plotly_chart(fig3)

def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    df_input = pd.DataFrame([data])
    return preprocess_data(df_input, for_prediction=True)

if page == "Prediction":
    st.header("Make a Prediction")
    st.write("Enter passenger details to predict survival probability.")
    st.markdown("""
    - **Passenger Class**: 1 = First, 2 = Second, 3 = Third
    - **Sex**: Passenger's gender
    - **Age**: Passenger's age (0-100)
    - **Siblings/Spouses Aboard**: Number of siblings or spouses
    - **Parents/Children Aboard**: Number of parents or children
    - **Fare**: Ticket price
    - **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
    """)
    with st.form("prediction_form"):
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ['male', 'female'])
        age = st.slider("Age", 0, 100, 30)
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
        embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])
        submitted = st.form_submit_button("Predict")

        if submitted:
            with st.spinner("Making prediction..."):
                try:
                    if age < 0 or fare < 0:
                        st.error("Age and Fare must be non-negative.")
                    else:
                        input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
                        prediction = model.predict(input_data)[0]
                        probability = model.predict_proba(input_data)[0]
                        st.write(f"Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
                        st.write(f"Survival Probability: {probability[1]:.2%}")
                except Exception as e:
                    st.error(f"Error: {e}")

@st.cache_data
def compute_performance_metrics():
    X = preprocess_data(df)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }

if page == "Model Performance":
    st.header("Model Performance")
    metrics = compute_performance_metrics()
    
    st.subheader("Metrics")
    st.write(f"Accuracy: {metrics['accuracy']:.4f}")
    st.text("Classification Report:")
    st.code(metrics['report'])

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)
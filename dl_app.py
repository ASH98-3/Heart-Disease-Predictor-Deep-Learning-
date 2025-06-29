import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("heart_disease_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

df = pd.read_csv("heartkaggle_copy.csv")


X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_test_scaled = scaler.transform(X_test)


y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)  
accuracy = accuracy_score(y_test, y_pred)

st.title("Heart Disease Prediction App")


st.subheader("Model Performance")
st.write(f"**Model Accuracy:** {accuracy:.2%}")


st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()), int(df.age.mean()))
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.sidebar.slider("Chest Pain Type (0-3)", int(df.cp.min()), int(df.cp.max()), int(df.cp.mean()))
trestbps = st.sidebar.slider("Resting Blood Pressure", int(df.trestbps.min()), int(df.trestbps.max()), int(df.trestbps.mean()))
chol = st.sidebar.slider("Cholesterol Level", int(df.chol.min()), int(df.chol.max()), int(df.chol.mean()))
thalach = st.sidebar.slider("Max Heart Rate Achieved", int(df.thalach.min()), int(df.thalach.max()), int(df.thalach.mean()))
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.sidebar.slider("ST Depression", float(df.oldpeak.min()), float(df.oldpeak.max()), float(df.oldpeak.mean()))
slope = st.sidebar.slider("ST Slope (0-2)", int(df.slope.min()), int(df.slope.max()), int(df.slope.mean()))
ca = st.sidebar.slider("Number of Major Vessels (0-3)", int(df.ca.min()), int(df.ca.max()), int(df.ca.mean()))
thal = st.sidebar.slider("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", 1, 3, 2)


user_input = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})


user_input_scaled = scaler.transform(user_input)

if st.sidebar.button("Predict"):
    prediction = model.predict(user_input_scaled)
    result = "High risk of Heart Disease" if prediction[0][0] > 0.5 else "Low risk of Heart Disease"
    st.write(f"### Prediction: {result}")

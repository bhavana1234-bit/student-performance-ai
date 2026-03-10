import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------
# Page Settings
# -------------------
st.set_page_config(page_title="Student Score Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's exam score based on study habits and lifestyle factors.")

# -------------------
# Load Dataset
# -------------------
data = pd.read_csv("StudentPerformanceFactors.csv")

# Clean column names (removes spaces issues)
data.columns = data.columns.str.strip().str.replace(" ", "_")

# -------------------
# Features
# -------------------
X = data[['Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']]
y = data['Exam_Score']

# -------------------
# Train Model
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.sidebar.write(f"Model Accuracy: {accuracy:.2f}")

# -------------------
# User Inputs
# -------------------
st.header("Enter Student Details")

sleep = st.slider("Sleep Hours", 0, 12, 7)
previous = st.slider("Previous Scores", 0, 100, 50)
tutoring = st.slider("Tutoring Sessions", 0, 10, 2)
activity = st.slider("Physical Activity", 0, 10, 3)

# -------------------
# Prediction
# -------------------
if st.button("Predict Exam Score"):

    input_data = pd.DataFrame(
        [[sleep, previous, tutoring, activity]],
        columns=['Sleep_Hours','Previous_Scores','Tutoring_Sessions','Physical_Activity']
    )

    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")
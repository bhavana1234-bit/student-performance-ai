import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --------------------------------
# Page Setup
# --------------------------------
st.set_page_config(page_title="Student Performance AI Dashboard", layout="wide")

st.title("🎓 Student Performance Prediction Dashboard")
st.write("Predict a student's exam score based on study habits and lifestyle factors.")

# --------------------------------
# Load Dataset
# --------------------------------
data = pd.read_csv("StudentPerformanceFactors.csv")

data.columns = data.columns.str.strip().str.replace(" ", "_")

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.title("🎓 Student AI Dashboard")

st.sidebar.info(
"""
This AI app predicts student exam scores based on key lifestyle factors.

Features
✔ ML Prediction  
✔ Data Visualization  
✔ Study Recommendations  
"""
)

st.sidebar.write("Created by **Bhavana**")

# --------------------------------
# Encode text columns
# --------------------------------
encoder = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = encoder.fit_transform(data[col])

# --------------------------------
# Selected Features (ONLY 5)
# --------------------------------
features = [
    "Attendance",
    "Sleep_Hours",
    "Study_Hours",
    "Previous_Scores",
    "Physical_Activity"
]

target = "Exam_Score"

X = data[features]
y = data[target]

# --------------------------------
# Train Model
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = r2_score(y_test, preds)

st.sidebar.write(f"Model Accuracy: {accuracy:.2f}")

# --------------------------------
# Dataset Overview
# --------------------------------
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(data))
col2.metric("Average Score", round(data["Exam_Score"].mean(), 2))
col3.metric("Highest Score", data["Exam_Score"].max())

# --------------------------------
# Prediction Section
# --------------------------------
st.header("🎯 Predict Exam Score")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    sleep = st.slider("Sleep Hours", 0, 12, 7)
    study = st.slider("Study Hours", 0, 10, 3)

with col2:
    previous = st.slider("Previous Score", 0, 100, 60)
    activity = st.slider("Physical Activity (hrs/week)", 0, 10, 3)

input_data = pd.DataFrame({
    "Attendance": [attendance],
    "Sleep_Hours": [sleep],
    "Study_Hours": [study],
    "Previous_Scores": [previous],
    "Physical_Activity": [activity]
})

# --------------------------------
# Prediction Button
# --------------------------------
if st.button("Predict Score"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")

# --------------------------------
# Heatmap
# --------------------------------
st.header("📊 Correlation Heatmap")

numeric_data = data.select_dtypes(include=["number"])

corr = numeric_data.corr()

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

plt.xticks(rotation=45)
plt.yticks(rotation=0)

st.pyplot(fig)

# --------------------------------
# Score Distribution
# --------------------------------
st.header("📈 Exam Score Distribution")

fig2, ax2 = plt.subplots()

sns.histplot(data["Exam_Score"], bins=20, kde=True, ax=ax2)

st.pyplot(fig2)

# --------------------------------
# Dataset Viewer
# --------------------------------
st.header("📂 Dataset Explorer")

if st.checkbox("Show Dataset"):
    st.dataframe(data)

# --------------------------------
# Download Dataset
# --------------------------------
st.header("⬇ Download Dataset")

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="student_data.csv",
    mime="text/csv"
)
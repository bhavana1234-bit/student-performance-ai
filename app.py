import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --------------------------------
# Page Setup
# --------------------------------
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance Prediction")
st.write("Predict exam score using study habits.")

# --------------------------------
# Load Dataset
# --------------------------------
data = pd.read_csv("StudentPerformanceFactors.csv")

# Clean column names
data.columns = data.columns.str.strip().str.replace(" ", "_")

# Show columns (for debugging if needed)
# st.write(data.columns)

# --------------------------------
# Detect Available Features
# --------------------------------
possible_features = [
    "Attendance",
    "Sleep_Hours",
    "Study_Hours",
    "Previous_Scores",
    "Physical_Activity"
]

features = [col for col in possible_features if col in data.columns]

target = "Exam_Score"

if target not in data.columns:
    st.error("❌ 'Exam_Score' column not found in dataset.")
    st.stop()

# --------------------------------
# Train Model
# --------------------------------
X = data[features]
y = data[target]

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
col2.metric("Average Score", round(data[target].mean(), 2))
col3.metric("Highest Score", data[target].max())

# --------------------------------
# Prediction Section
# --------------------------------
st.header("🎯 Predict Exam Score")

input_data = {}

for feature in features:
    input_data[feature] = st.slider(feature, 0, 100, 50)

input_df = pd.DataFrame([input_data])

if st.button("Predict Score"):

    prediction = model.predict(input_df)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")

# --------------------------------
# Correlation Heatmap
# --------------------------------
st.header("📊 Correlation Heatmap")

numeric_data = data.select_dtypes(include="number")

corr = numeric_data.corr()

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)

# --------------------------------
# Score Distribution
# --------------------------------
st.header("📈 Exam Score Distribution")

fig2, ax2 = plt.subplots()

sns.histplot(data[target], bins=20, kde=True, ax=ax2)

st.pyplot(fig2)

# --------------------------------
# Dataset Viewer
# --------------------------------
st.header("📂 Dataset")

if st.checkbox("Show Dataset"):
    st.dataframe(data)
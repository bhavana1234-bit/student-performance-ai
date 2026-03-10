import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Student Performance AI Dashboard", layout="wide")

st.title("🎓 Student Performance Prediction Dashboard")
st.write("Predict a student's exam score based on study habits and lifestyle factors.")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("StudentPerformanceFactors.csv")

# Clean column names
data.columns = data.columns.str.strip().str.replace(" ", "_")

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoder = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = encoder.fit_transform(data[col])

# -----------------------------
# Define Target and Features
# -----------------------------
target = "Exam_Score"

X = data.drop(columns=[target])
y = data[target]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    scores[name] = r2_score(y_test, preds)

# Best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

# -----------------------------
# Sidebar Model Scores
# -----------------------------
st.sidebar.header("⚙ Model Performance")

for name, score in scores.items():
    st.sidebar.write(f"{name}: {score:.2f}")

st.sidebar.success(f"Best Model: {best_model_name}")

# -----------------------------
# Model Comparison Chart
# -----------------------------
st.header("📊 Model Accuracy Comparison")

fig, ax = plt.subplots()

ax.bar(scores.keys(), scores.values())
ax.set_ylabel("R² Score")
ax.set_title("Model Performance")

st.pyplot(fig)

# -----------------------------
# Prediction Section
# -----------------------------
st.header("🎯 Predict Exam Score")

input_data = {}

cols = st.columns(3)

for i, feature in enumerate(X.columns):
    with cols[i % 3]:
        input_data[feature] = st.number_input(feature, float(X[feature].mean()))

input_df = pd.DataFrame([input_data])

# -----------------------------
# AI Recommendation Function
# -----------------------------
def generate_recommendations(data):

    recs = []

    if data["Sleep_Hours"][0] < 6:
        recs.append("💤 Try sleeping 7-8 hours for better concentration.")

    if data["Tutoring_Sessions"][0] < 2:
        recs.append("👨‍🏫 Attending more tutoring sessions may improve performance.")

    if data["Physical_Activity"][0] < 2:
        recs.append("🏃 Physical activity helps improve focus.")

    if data["Previous_Scores"][0] < 50:
        recs.append("📚 Increase study time to improve previous scores.")

    if len(recs) == 0:
        recs.append("🎉 Great habits! Keep maintaining them.")

    return recs

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Score"):

    prediction = best_model.predict(input_df)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")

    st.subheader("🤖 AI Study Recommendations")

    for r in generate_recommendations(input_df):
        st.write(r)

# -----------------------------
# Clean Correlation Heatmap
# -----------------------------
st.header("📊 Correlation Heatmap")

numeric_data = data.select_dtypes(include=["number"])

corr = numeric_data.corr()

fig2, ax2 = plt.subplots(figsize=(10,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax2
)

plt.xticks(rotation=45)
plt.yticks(rotation=0)

st.pyplot(fig2)

# -----------------------------
# Feature Importance
# -----------------------------
if hasattr(best_model, "feature_importances_"):

    st.header("📈 Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10,5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance,
        ax=ax3
    )

    st.pyplot(fig3)

# -----------------------------
# AI Insights
# -----------------------------
st.header("🤖 Dataset Insights")

top_feature = corr["Exam_Score"].abs().sort_values(ascending=False).index[1]

st.info(f"Feature most related to exam score: {top_feature}")

st.write(f"Average exam score in dataset: {data['Exam_Score'].mean():.2f}")

# -----------------------------
# Dataset Viewer
# -----------------------------
st.header("📂 Dataset Explorer")

if st.checkbox("Show Dataset"):
    st.dataframe(data)
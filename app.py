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

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="AI Student Performance Dashboard", layout="wide")

st.title("🎓 AI Student Performance Dashboard")
st.write("Analyze study habits and predict exam scores using Machine Learning.")

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("StudentPerformanceFactors.csv")

# Clean column names
data.columns = data.columns.str.strip().str.replace(" ", "_")

# -------------------------
# Encode Categorical Data
# -------------------------
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = label_encoder.fit_transform(data[column])

# -------------------------
# Features and Target
# -------------------------
target = "Exam_Score"

X = data.drop(columns=[target])
y = data[target]

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train Models
# -------------------------
lr = LinearRegression()
rf = RandomForestRegressor()
dt = DecisionTreeRegressor()
gb = GradientBoostingRegressor()

models = {
    "Linear Regression": lr,
    "Random Forest": rf,
    "Decision Tree": dt,
    "Gradient Boosting": gb
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    scores[name] = r2_score(y_test, preds)

# -------------------------
# Sidebar Model Performance
# -------------------------
st.sidebar.header("⚙ Model Performance")

for model, score in scores.items():
    st.sidebar.write(f"{model}: {score:.2f}")

best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

st.sidebar.success(f"🏆 Best Model: {best_model_name}")

# -------------------------
# Model Comparison Chart
# -------------------------
st.header("📊 Model Accuracy Comparison")

fig, ax = plt.subplots()

ax.bar(scores.keys(), scores.values())
ax.set_ylabel("R² Score")
ax.set_title("Machine Learning Model Performance")

st.pyplot(fig)

# -------------------------
# Prediction Section
# -------------------------
st.header("🎯 Predict Student Exam Score")

input_data = {}

cols = st.columns(3)

for i, feature in enumerate(X.columns):
    with cols[i % 3]:
        input_data[feature] = st.number_input(feature, value=float(X[feature].mean()))

input_df = pd.DataFrame([input_data])

# -------------------------
# AI Recommendation Function
# -------------------------
def generate_recommendations(input_data):

    recs = []

    if input_data["Sleep_Hours"][0] < 6:
        recs.append("💤 Try sleeping at least 7–8 hours for better focus.")

    if input_data["Previous_Scores"][0] < 50:
        recs.append("📚 Previous scores are low. Increase study hours.")

    if input_data["Tutoring_Sessions"][0] < 2:
        recs.append("👨‍🏫 Consider attending more tutoring sessions.")

    if input_data["Physical_Activity"][0] < 2:
        recs.append("🏃 Physical activity helps improve concentration.")

    if len(recs) == 0:
        recs.append("🎉 Excellent habits! Keep maintaining this routine.")

    return recs

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict Score"):

    prediction = best_model.predict(input_df)

    st.success(f"🎯 Predicted Exam Score: {prediction[0]:.2f}")

    st.subheader("🤖 AI Study Recommendations")

    recommendations = generate_recommendations(input_df)

    for r in recommendations:
        st.write(r)

# -------------------------
# Correlation Heatmap
# -------------------------
st.header("📊 Correlation Heatmap")

numeric_data = data.select_dtypes(include=['number'])

fig2, ax2 = plt.subplots()

sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax2)

st.pyplot(fig2)

# -------------------------
# Feature Importance
# -------------------------
if hasattr(best_model, "feature_importances_"):

    st.header("📈 Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig3, ax3 = plt.subplots()

    ax3.bar(importance["Feature"], importance["Importance"])
    ax3.set_xticklabels(importance["Feature"], rotation=45)

    st.pyplot(fig3)

# -------------------------
# AI Insights
# -------------------------
st.header("🤖 AI Insights")

top_feature = numeric_data.corr()["Exam_Score"].abs().sort_values(ascending=False).index[1]

st.info(f"📌 The feature most correlated with exam score is **{top_feature}**.")

avg_score = data["Exam_Score"].mean()

st.write(f"📊 Average Exam Score in dataset: **{avg_score:.2f}**")

# -------------------------
# Dataset Explorer
# -------------------------
st.header("📂 Dataset Explorer")

if st.checkbox("Show Dataset"):
    st.dataframe(data)
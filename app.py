import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Student AI Dashboard", layout="wide")

# ----------------------------
# Custom UI Styling
# ----------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

.big-title{
font-size:42px;
font-weight:bold;
color:#2c3e50;
}

.prediction-card{
padding:25px;
border-radius:12px;
background:white;
box-shadow:0px 6px 20px rgba(0,0,0,0.15);
text-align:center;
}

.metric-card{
background:white;
padding:15px;
border-radius:10px;
box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.markdown('<p class="big-title">🎓 Student Performance AI Dashboard</p>', unsafe_allow_html=True)
st.write("Predict exam score using lifestyle and study habits.")

# ----------------------------
# Load Data
# ----------------------------
data = pd.read_csv("StudentPerformanceFactors.csv")
data.columns = data.columns.str.strip().str.replace(" ","_")

# ----------------------------
# Features
# ----------------------------
# ----------------------------
# Features (Safe Selection)
# ----------------------------

possible_features = [
    "Attendance",
    "Sleep_Hours",
    "Hours_Studied",
    "Previous_Scores",
    "Physical_Activity"
]

# Only keep columns that exist in dataset
features = [f for f in possible_features if f in data.columns]

target = "Exam_Score"

# ----------------------------
# Train Model
# ----------------------------
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

preds = model.predict(X_test)

accuracy = r2_score(y_test,preds)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙ Model Info")
st.sidebar.metric("Model Accuracy", f"{accuracy:.2f}")
st.sidebar.write("Model: Linear Regression")

# ----------------------------
# Dataset Stats
# ----------------------------
st.header("📊 Dataset Overview")

c1,c2,c3 = st.columns(3)

c1.metric("Total Students",len(data))
c2.metric("Average Score",round(data[target].mean(),2))
c3.metric("Highest Score",data[target].max())

# ----------------------------
# Prediction Section
# ----------------------------
st.header("🎯 Predict Exam Score")

col1,col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance (%)",0,100,70)
    sleep = st.slider("Sleep Hours",0,12,7)
    study = st.slider("Study Hours",0,10,3)

with col2:
    previous = st.slider("Previous Score",0,100,60)
    activity = st.slider("Physical Activity",0,10,3)

input_data = pd.DataFrame({
"Attendance":[attendance],
"Sleep_Hours":[sleep],
"Study_Hours":[study],
"Previous_Scores":[previous],
"Physical_Activity":[activity]
})

# ----------------------------
# Prediction
# ----------------------------
if st.button("🚀 Predict Score"):

    score = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="prediction-card">
        <h2>Predicted Exam Score</h2>
        <h1 style="color:#27ae60;">{score:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

    if score >= 80:
        st.success("Excellent performance expected 🎉")

    elif score >= 60:
        st.info("Good performance expected 👍")

    else:
        st.warning("Needs improvement. Study more 📚")

# ----------------------------
# Visualizations
# ----------------------------
st.header("📊 Data Visualizations")

col1,col2 = st.columns(2)

# Heatmap
with col1:

    numeric = data.select_dtypes(include="number")

    corr = numeric.corr()

    fig,ax = plt.subplots(figsize=(6,4))

    sns.heatmap(corr,annot=True,cmap="coolwarm",ax=ax)

    st.pyplot(fig)

# Score distribution
with col2:

    fig2,ax2 = plt.subplots()

    sns.histplot(data[target],bins=20,kde=True,ax=ax2)

    ax2.set_title("Exam Score Distribution")

    st.pyplot(fig2)

# ----------------------------
# Dataset Viewer
# ----------------------------
st.header("📂 Dataset")

if st.checkbox("Show Dataset"):
    st.dataframe(data)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.write("Built with Python, Machine Learning and Streamlit")
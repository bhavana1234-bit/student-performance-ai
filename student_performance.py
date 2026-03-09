import pandas as pd
data=pd.read_csv("StudentPerformanceFactors.csv")
print(data.head())
print(data.info())
print(data.describe())
data = data.dropna()
data = data.fillna(data.mean(numeric_only=True))
print(data.isnull().sum())
X = data[['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores','Tutoring_Sessions','Physical_Activity']]
y = data['Exam_Score']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(predictions[:5])
from sklearn.metrics import r2_score
score = r2_score(y_test,predictions)
print("Model Accuracy:",score)
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=data)
plt.title("Hours Studied vs Exam Score")
plt.show()
sns.scatterplot(x='Attendance', y='Exam_Score', data=data)
plt.title("Attendance vs Exam Score")
plt.show()
sns.scatterplot(x='Previous_Scores', y='Exam_Score', data=data)
plt.title("Previous Scores vs Final Exam Score")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()
sample = [[6, 85, 7, 70, 2, 3]]

predicted_score = model.predict(sample)

print("Predicted Exam Score:", predicted_score)
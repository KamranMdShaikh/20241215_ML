import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt  # Correct import

# Load dataset
df = pd.read_csv('diabetes.csv')
print(df.head())
print(df.shape)
print(df.columns)

# Dataframe with X and Y values
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy scores and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the confusion matrix using seaborn
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


### Classification report 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
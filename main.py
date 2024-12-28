## Library

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

## Load dataset
df = pd.read_csv('diabetes.csv')
print(df.head())
print(df.shape)
print(df.columns)

## Dataframe with X and Y values
X = df.drop('Outcome',axis=1)
# print(X.head())
y = df['Outcome']
# print(y.head())

## Train Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


## accuracy scores and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Score:", confusion_matrix(y_test, y_pred))


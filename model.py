# Title: Rain Prediction Model
# Author: Karthik Prasad
# If any one are using this model then after model evaluation include new data into your directory
# The data set that I used for training was based on ranfall in Oceanic regions

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, jaccard_score, f1_score, log_loss
import warnings
warnings.filterwarnings("ignore")

# Load and Preprocess Data
df = pd.read_csv("Weather_Data.csv")
df = pd.get_dummies(df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df.replace(['No', 'Yes'], [0, 1], inplace=True)
df.drop('Date', axis=1, inplace=True)
df = df.astype(float)
features = df.drop(columns='RainTomorrow', axis=1)
Y = df['RainTomorrow']

# Split Data for Training and Testing
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

# Train and Evaluate Models

# Linear Regression
LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)
predictions = LinearReg.predict(x_test)
LinearRegression_MAE = mean_absolute_error(y_test, predictions)
LinearRegression_MSE = mean_squared_error(y_test, predictions)
LinearRegression_R2 = r2_score(y_test, predictions)

# K-Nearest Neighbors (KNN)
KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(x_train, y_train)
predictions = KNN.predict(x_test)
KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

# Decision Tree
Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)
predictions = Tree.predict(x_test)
Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)

# Logistic Regression
LR = LogisticRegression(solver='liblinear')
LR.fit(x_train, y_train)
predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)
LR_Accuracy_Score = accuracy_score(y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predict_proba)

# Support Vector Machine (SVM)
SVM = svm.SVC(probability=True)
SVM.fit(x_train, y_train)
predictions = SVM.predict(x_test)
SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)

# Report Results
report = pd.DataFrame({
    'Model': ['Linear Regression', 'KNN', 'Decision Tree', 'Logistic Regression', 'SVM'],
    'Accuracy': [None, KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score],
    'Jaccard Index': [None, KNN_JaccardIndex, Tree_JaccardIndex, LR_JaccardIndex, SVM_JaccardIndex],
    'F1 Score': [None, KNN_F1_Score, Tree_F1_Score, LR_F1_Score, SVM_F1_Score],
    'Log Loss': [None, None, None, LR_Log_Loss, None],
    'MAE': [LinearRegression_MAE, None, None, None, None],
    'MSE': [LinearRegression_MSE, None, None, None, None],
    'R2 Score': [LinearRegression_R2, None, None, None, None]
})
print(report)
# From this report we can choose best model with highest accuracy

# Prediction on New Data
def preprocess_new_data(new_data_path, features_columns):
    new_data = pd.read_csv(new_data_path)
    new_data = pd.get_dummies(new_data, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
    new_data.replace(['No', 'Yes'], [0, 1], inplace=True)
    new_data.drop('Date', axis=1, inplace=True)
    new_data = new_data.astype(float)
    missing_cols = set(features_columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[features_columns]
    return new_data

def predict_rain(model, new_data_path, features_columns):
    new_data = preprocess_new_data(new_data_path, features_columns)
    predictions = model.predict(new_data)
    predictions_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]
    new_data['RainTomorrowPrediction'] = predictions_labels
    return new_data[['RainTomorrowPrediction']]

features_columns = features.columns
predicted_data = predict_rain(LR, "new_weather_data.csv", features_columns)
print(predicted_data.head())

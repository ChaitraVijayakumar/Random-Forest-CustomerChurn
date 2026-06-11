import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

import pickle

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv('customerDataset.csv')

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.head()

# Basic information
print(df.info())

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fix TotalCharges column (it contains blank strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values in TotalCharges with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID (not useful)
df.drop('customerID', axis=1, inplace=True)

print("Shape after cleaning:", df.shape)

print("Churn Rate:\n", df['Churn'].value_counts(normalize=True)*100)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Categorical Columns:", categorical_cols)

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    if col != 'Churn':                    
        df[col] = le.fit_transform(df[col])

# Encode Target Variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("Final Columns:", df.columns.tolist())
print("Churn Value Counts:", df['Churn'].value_counts())

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# Random Forest (Best Model)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Decision Tree

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
dt_model = DecisionTreeRegressor(
    max_depth=5,           # Prevents overfitting
    random_state=42
)

dt_model.fit(X_train, y_train)

def evaluate_model(model, name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, pred)*100:.2f}%")
    print(f"ROC-AUC  : {roc_auc_score(y_test, prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred))

evaluate_model(rf_model, "Random Forest")


feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance.head(10))

pickle.dump(rf_model, open("model.pkl", "wb"))
print("(Random Forest) saved as model.pkl")
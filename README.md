# Random-Forest-Customer_Churn
# Customer Churn Prediction using Machine Learning

**Predict whether a customer will leave the company using Random Forest Classifier**

## Problem Statement

In the highly competitive telecom industry, **customer retention** is more important than acquisition. This project aims to predict which customers are likely to **churn** (leave the company) so that the business can take proactive retention strategies.

##  Key Features

- Comprehensive **Exploratory Data Analysis (EDA)**
- Proper handling of missing values and categorical encoding
- Training of **3 models**: Logistic Regression, Decision Tree & Random Forest
- Detailed model comparison and evaluation
- **Feature Importance** analysis for business insights
- Saved model for future predictions

## Technologies Used

- **Python**
- **Pandas & NumPy** – Data Manipulation
- **Matplotlib & Seaborn** – Data Visualization
- **Scikit-Learn** – Machine Learning
- **Pickle** – Model Saving

## Project Structure

Customer-Churn-Prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebook.ipynb
├── model.pkl
├── README.md
├── requirements.txt
└── images/
    ├── eda_plots.png
    └── feature_importance.png

## How to Run the Project

Clone the repository:git clone https://github.com/YOUR-USERNAME/Customer-Churn-Prediction.git
Install dependencies: pip install -r requirements.txt
Open the notebook: jupyter notebook
Run it

## Business Recommendations
Based on the model:

Offer long-term contracts with discounts
Provide better tech support to high-risk customers
Review pricing strategy for high monthly charges
Target new customers with onboarding offers


## Future Improvements

Hyperparameter tuning using GridSearchCV
Feature Engineering (e.g., creating "Total Services" feature)
Customer Segmentation
Build a Streamlit Web App for real-time predictions
Deploy model using Flask/FastAPI

Made with ❤️ for learning and showcasing Machine Learning skills
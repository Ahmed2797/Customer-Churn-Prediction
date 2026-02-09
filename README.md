# Customer-Churn-Prediction

Customer Churn Prediction System is an end-to-end machine learning project designed to identify customers who are likely to stop using a service. The system analyzes historical customer behavior and service usage patterns to predict churn probability and support data-driven retention strategies.

## 📌 Project Overview

Bank Churn Modelling is a **machine learning project** aimed at predicting whether a customer will leave (churn) a bank or stay. By analyzing historical customer data, the model helps banks **identify high-risk customers** and take proactive measures to improve retention.

## 📂 Dataset

    https://drive.google.com/file/d/1hA0hdGr_mzWsrnRgQkzN5uwYmht7l3bC/view?usp=sharing
    

The dataset contains customer information and bank-related features. Common features include:

| Feature         | Description                                         |
| --------------- | --------------------------------------------------- |
| CustomerId      | Unique ID of the customer                           |
| Surname         | Customer's last name                                |
| CreditScore     | Credit score of the customer                        |
| Geography       | Country of the customer                             |
| Gender          | Male/Female                                         |
| Age             | Customer age                                        |
| Tenure          | Number of years with the bank                       |
| Balance         | Account balance                                     |
| NumOfProducts   | Number of bank products used                        |
| HasCrCard       | Whether the customer has a credit card (1/0)        |
| IsActiveMember  | Whether the customer is active (1/0)                |
| EstimatedSalary | Estimated yearly salary                             |
| Exited          | Target variable: 1 if customer churned, 0 otherwise |

## 🎯 Objective

* Predict customer churn using machine learning algorithms.
* Identify important features influencing churn.
* Provide actionable insights for the bank to reduce churn.

---

## 🛠️ Methodology

1. **Data Exploration & Visualization**

   * Check for missing values, distributions, and correlations.
   * Visualize churn patterns using plots.

2. **Data Preprocessing**

   * Encode categorical variables (e.g., Gender, Geography).
   * Scale numerical features.
   * Split data into training and test sets.

3. **Modeling**

   * Algorithms used:

     * Logistic Regression
     * Decision Tree
     * Random Forest
     * XGBoost / Gradient Boosting
   * Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

4. **Evaluation Metrics**

   * Accuracy
   * Precision, Recall, F1-Score
   * ROC-AUC score

5. **Feature Importance**

   * Identify key features contributing to churn prediction.

---

## ⚙️ Installation

Clone the repository and install dependencies:

    # Clone the repository
    git clone https://github.com/Ahmed2797/Customer-Churn-Prediction.git
    cd bank-churn-modelling

    # Create and activate a conda environment
    conda create -n ml python=3.12
    conda activate ml

    # setup
    python setup.py install

    # Install dependencies
    pip install -r requirements.txt
    
    # Push data Mongo
    python push_data_mongo.py

---

## 🏃 How to Run

1. Launch Jupyter Notebook or Python scripts.
2. Follow the notebooks for:

   * Data exploration
   * Preprocessing
   * Model training and evaluation
3. Visualize results and feature importance.

---

## 📈 Key Features

* Clean and modular **end-to-end ML pipeline**
* Multiple algorithms for prediction
* Hyperparameter tuning for optimal performance
* Feature importance insights for business decisions
* Ready for **deployment / integration**

---

## 📬 Contact

**Author:** github.com/Ahmed2797

**Interest:** Machine Learning, Data Science, Predictive Modelling

---

⭐ If you find this project helpful, give it a star on GitHub!

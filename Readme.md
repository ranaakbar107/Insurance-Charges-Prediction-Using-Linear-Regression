# 💰 Insurance Charges Prediction | ML Pipeline + Log Transformation

## 📌 Overview

This project predicts medical insurance charges based on individual attributes such as age, BMI, smoking status, and region using Machine Learning.

The project follows an end-to-end workflow including Exploratory Data Analysis (EDA), feature preprocessing, and an industry-level ML pipeline with log transformation.

---

## 🎯 Problem Statement

Insurance companies need to estimate medical costs for individuals based on various factors.
The objective of this project is to build a regression model that accurately predicts insurance charges and identifies the most influential features.

---

## 📊 Dataset

* **Source:** Kaggle – Medical Cost Personal Dataset
* **Rows:** 1338
* **Columns:** 7

### 🔹 Features

* age
* sex
* bmi
* children
* smoker
* region

### 🎯 Target

* charges (medical insurance cost)

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights derived from the data:

* Charges distribution is **right-skewed**, indicating presence of high-cost outliers
* **Smoking status** has the strongest impact on charges
* **Age** shows a moderate positive correlation
* **BMI** has a weaker but noticeable relationship
* Features like **sex, region, and children** have minimal influence

---

## ⚙️ Data Preprocessing

* Categorical variables handled using **One-Hot Encoding**
* Numerical features scaled using **StandardScaler**
* Used **ColumnTransformer** for structured preprocessing
* Applied **train-test split** to avoid data leakage

---

## 🤖 Model & Pipeline

* Model used: **Linear Regression**
* Built using **Scikit-learn Pipeline**
* Integrated preprocessing + model in a single pipeline
* Applied **TransformedTargetRegressor** for log transformation

### 🔄 Pipeline Flow

Raw Data → Preprocessing → Model → Log Transformation → Prediction

---

## 📈 Model Evaluation

* **R² Score:** (0.865231697953168)
* **MAE:** (2757.759204250125)
* **RMSE:** (4574.123734451315)

The model performance improves significantly after applying log transformation to the target variable.

---

## 📊 Visualizations

The project includes:

* Distribution plots (charges)
* Boxplots (Smoker vs Charges)
* Scatter plots (Age/BMI vs Charges)
* Correlation heatmap

---

## 📂 Project Structure

```
insurance-charges-prediction/
│
├── notebook.ipynb
├── requirements.txt
├── README.md
└── data/
```

---

## 🚀 Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook

---

## 🔮 Future Improvements

* Apply advanced models (Ridge, Lasso)
* Perform hyperparameter tuning
* Build a Streamlit web application
* Deploy model as an API

---

## 👤 Author

**Muhammad Akbar**
Machine Learning Enthusiast

---

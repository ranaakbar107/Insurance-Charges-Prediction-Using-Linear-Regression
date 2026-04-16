#!/usr/bin/env python
# coding: utf-8

# # Insurance Charges Prediction using Linear Regression

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model


# # DATA ACQUISITION & INSPECTION

# In[ ]:


path = r"Data\insurance.csv"
df = pd.read_csv(path)
df.head()


# # Data Loading and Initial Preview

# In[ ]:


df.info()


# # Metadata and Structure Inspection

# In[ ]:


df.describe()


# # Checking Columns

# In[ ]:


df.columns
df.dtypes


# # Data Validation

# In[ ]:


df.isnull().sum()


# In[ ]:


df['charges']


# # Exploratory Data Analysis (EDA)

# In[ ]:


plt.figure(figsize=(8, 5))
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of Charges")
plt.show()


# In[ ]:


sns.boxplot(x = 'smoker', y = 'charges', data=df)
plt.title("Smoker VS Charges")
plt.show()


# In[ ]:


sns.heatmap(df.corr(numeric_only=True), vmin = -1, vmax = 1, annot=True, cmap='RdBu')
plt.title('Heatmap')
plt.show()


# In[ ]:


sns.scatterplot(x = 'bmi', y = 'charges', data = df)
plt.title('BMI VS Charges')
plt.show()


# In[ ]:


sns.scatterplot(x = 'age', y = 'charges', data = df)
plt.title('Age VS Charges')
plt.show()


# In[ ]:


sns.boxplot(x = 'sex', y = 'charges', data = df)
plt.title('Sex VS Charges')
plt.show()


# In[ ]:


sns.boxplot(x = 'region', y = 'charges', data = df)
plt.title('Region VS Charges')
plt.show()


# In[ ]:


sns.boxplot(x = 'children', y = 'charges', data = df)
plt.title('Children VS Charges')
plt.show()


# In[ ]:


sns.scatterplot(x = 'children', y = 'charges', data = df)
plt.title('Children VS Charges')
plt.show()


# # FEATURE ENGINEERING

# In[ ]:


df["age_smoker"] = df["age"] * (df["smoker"] == "yes").astype(int)
df["bmi_smoker"] = df["bmi"] * (df["smoker"] == "yes").astype(int)


# # Features and Target Defining

# In[ ]:


X = df.drop("charges", axis=1)
y = df["charges"]


# # Identifying Numeric aur Categorical Columns

# In[ ]:


numeric_features = ["age", "bmi", "children", "age_smoker", "bmi_smoker"]
categorical_features = ["sex", "smoker", "region"]


# # Train-test Split

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# # Preprocessing Pipelines

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])


# # ColumnTransformer

# In[ ]:


preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# # Full Model Pipeline

# In[ ]:


from sklearn.linear_model import LinearRegression

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])


# In[ ]:


from sklearn.compose import TransformedTargetRegressor
import numpy as np

model = TransformedTargetRegressor(
    regressor=model_pipeline,
    func=np.log,
    inverse_func=np.exp
)


# # Model Training

# In[ ]:


model_pipeline.fit(X_train, y_train)


# # Predictions

# In[ ]:


y_pred = model_pipeline.predict(X_test)

print(y_pred[:5])


# # Evaluation

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# # Actual vs Predicted Plot

# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.show()


# # Residual Check

# In[ ]:


residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# # Entire Pipeline Save

# In[ ]:


import pickle

with open("insurance_linear_regression_pipeline.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script *.ipynb')


# In[ ]:


get_ipython().system('pipreqs . --force')


# In[ ]:


with open(".gitignore", "w") as f:
    f.write("""# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Jupyter
.ipynb_checkpoints/

# Virtual Environment
venv/
env/

# Environment Variables
.env

# OS Files
.DS_Store
Thumbs.db
""")


# In[ ]:





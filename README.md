# House Price Prediction using Simple Regression Models

## Project Overview

This project builds a regression-based machine learning model to predict house prices using a public dataset from Kaggle.

The objective of this project is to demonstrate a structured end-to-end regression workflow by comparing:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  

The project focuses on building a clean modeling pipeline, evaluating model performance, and performing inference using the final trained model.

---

## Repository Structure

This repository contains the following notebooks:

- **01_dataset_loading.ipynb**  
  Data loading and initial inspection of dataset structure.

- **02_EDA.ipynb**  
  Exploratory Data Analysis, including distribution analysis, skewness evaluation, and correlation analysis.

- **03_Modelling.ipynb**  
  Feature preparation, model training, cross-validation, hyperparameter tuning, evaluation, and model saving.

- **04_Inference.ipynb**  
  Demonstration of prediction using the saved trained model.

Additionally:
- `data/` → contains the raw dataset  
- `tuned_lasso_model.pkl` → saved final trained model  
- `README.md` → project documentation  

---

## Project Workflow

### 1. Data Loading
- Imported dataset from Kaggle source  
- Inspected structure, data types, and summary statistics  

### 2. Exploratory Data Analysis (EDA)
- Analyzed numerical distributions  
- Evaluated skewness of numerical features  
- Examined correlation between features and target variable  

### 3. Feature Engineering & Modeling
- Split dataset into training and testing sets  
- Applied encoding for categorical variables  
- Applied scaling for numerical features  
- Built regression models using:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression  
- Evaluated models using MAE, RMSE, and R²  
- Performed cross-validation to assess model stability  
- Conducted hyperparameter tuning for Lasso using GridSearchCV  
- Selected tuned Lasso as the final model  

### 4. Model Saving
- Saved the trained model pipeline as `tuned_lasso_model.pkl`  
- Ensured preprocessing steps are included in the saved pipeline  

### 5. Inference
- Loaded saved model  
- Performed predictions on new house data  
- Validated logical consistency of predicted values  

---

## Libraries Used

The following Python libraries were used:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `scipy`
- `feature-engine`
- `pickle`

---

## Key Learning Points

- Regression models can effectively predict structured tabular data.
- Proper scaling and encoding are essential for linear models.
- Cross-validation improves reliability of model evaluation.
- Hyperparameter tuning enhances model robustness.
- Saving the full pipeline ensures deployment readiness.

---

## Model Usage (Inference Example)

```python
import pickle
import pandas as pd

with open("tuned_lasso_model.pkl", "rb") as f:
    model = pickle.load(f)

new_house = pd.DataFrame([{
    "area": 6000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 1,
    "prefarea": "yes",
    "furnishingstatus": "semi-furnished"
}])

prediction = model.predict(new_house)
print(f"Predicted House Price: {prediction[0]:,.0f}")
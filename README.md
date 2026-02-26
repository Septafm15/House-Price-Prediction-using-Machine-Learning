# House Price Prediction using Simple Regression Models

## Project Overview

This project builds a simple regression-based machine learning model to predict house prices using a public dataset from Kaggle.

The objective of this project is to demonstrate a complete end-to-end regression workflow by comparing:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  

The focus of this project is not only prediction accuracy, but also building a clean and structured machine learning pipeline.

---

## Dataset

- Source: Kaggle – Housing Prices Dataset  
- Total observations: 545  
- Target variable: `price`  
- Feature types: Numerical and Categorical  

The dataset includes structured features such as property size, number of bedrooms and bathrooms, parking capacity, furnishing status, and location-related indicators.

---

## Project Workflow

The project follows a structured machine learning process:

### 1. Data Exploration (EDA)
- Checked dataset structure and data types  
- Verified no missing values  
- Analyzed numerical distributions  
- Evaluated skewness  
- Assessed outliers using IQR method  

### 2. Feature Engineering
- Split features into numerical and categorical  
- Separated continuous and discrete numerical features  
- Applied encoding for categorical variables  
- Applied scaling based on skewness categories  

### 3. Preprocessing Pipeline
- Implemented `ColumnTransformer` for clean preprocessing  
- Used `Pipeline` to combine preprocessing and model  
- Prevented data leakage by fitting transformations only on training data  

### 4. Model Training
- Trained Linear Regression as baseline  
- Compared with Ridge and Lasso regression  
- Evaluated using MAE, RMSE, and R²  

### 5. Cross-Validation
- Applied 5-fold cross-validation  
- Assessed model stability and generalization  
- Selected best-performing model  

### 6. Hyperparameter Tuning
- Performed GridSearchCV for the best model  
- Optimized the parameter  
- Selected tuned model as final model  

### 7. Model Saving
- Saved final model as serialized pipeline  
- Ensured preprocessing steps are included  

### 8. Inference
- Demonstrated prediction on new house data  
- Validated logical consistency of predictions  

---

## Libraries Used

The following Python libraries were used in this project:

- **pandas** – data manipulation  
- **numpy** – numerical operations  
- **matplotlib** – data visualization  
- **seaborn** – data visualization 
- **scikit-learn** – modeling and preprocessing  
- **statsmodels** – regression diagnostics  
- **scipy** – statistical analysis  
- **feature-engine** – outlier handling  
- **pickle** – model serialization  
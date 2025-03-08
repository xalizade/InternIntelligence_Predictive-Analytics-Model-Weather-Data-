# InternIntelligence_Predictive-Analytics-Model-Weather-Data-

## Project Overview
This project aims to predict temperature values using historical weather data. It leverages various machine learning techniques and evaluates their performance through different metrics. The key steps involve data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation. Additionally, the project demonstrates the use of the Prophet model for time series forecasting.

## Table of Contents
1. Project Overview  
2. Dependencies  
3. Data Exploration  
4. Data Preprocessing  
5. Feature Engineering  
6. Model Selection & Training  
7. Hyperparameter Tuning  
8. Model Evaluation  
9. Data Visualization  
10. Conclusion  

## Dependencies
Ensure the following libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn fbprophet
```

## Data Exploration
The dataset is loaded from `weather_data.csv`. Initially, the dataset is explored to check for missing values and gain insights.
```python
import pandas as pd
data = pd.read_csv("weather_data.csv")
print(data.head())
print(data.isnull().sum())
```

## Data Preprocessing
Cleaning the data involves removing missing values and converting the date column to a datetime format.
```python
data.dropna(inplace=True)
data["date"] = pd.to_datetime(data["date"])
```

## Feature Engineering
New features are created from the existing date, temperature, and humidity columns. These include interactions, differences, and moving averages.
```python
data["temp_humidity_interaction"] = data["Temp_C"] * data["Rel Hum_%"]
data["temp_diff"] = data["Temp_C"].diff()
data["temp_avg_3"] = data["Temp_C"].rolling(window=3).mean()
```

## Model Selection & Training
Various regression models are explored, including Random Forest, Gradient Boosting, and Linear Regression, with performance evaluated using the R2 score.
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "LinearRegression": LinearRegression()
}
```

## Hyperparameter Tuning
Randomized Search Cross-Validation is used to optimize the hyperparameters for the Random Forest model.
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}
random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                                  param_distributions=param_dist,
                                  n_iter=5,
                                  cv=3,
                                  scoring="r2",
                                  n_jobs=-1)
```

## Model Evaluation
The final model is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R2 score.
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test_scaled)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

## Data Visualization
Using `matplotlib` and `seaborn`, key visualizations include:
- Feature Importance (Random Forest)
- Model Performance Comparison
- Actual vs Predicted Values
- Error Distribution
- Feature Correlation Heatmap

Example:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x=importances, y=feature_names)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Random Forest)")
plt.show()
```

## Conclusion
This project showcases the end-to-end process of forecasting temperature using machine learning models. It evaluates different models, highlights the most significant features, and demonstrates the effectiveness of hyperparameter tuning for performance improvement.


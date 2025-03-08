# InternIntelligence_Predictive-Analytics-Model-Weather-Data-

## Overview
This project focuses on analyzing historical weather data to predict temperature using Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn. The dataset includes temperature, humidity, and other weather-related attributes. The goal is to explore trends in temperature variations and develop an accurate forecasting model.

## Dataset
The dataset is stored in `weather_data.csv` and contains information such as:
- Date
- Temperature
- Humidity
- Wind Speed
- Atmospheric Pressure

## Features and Analysis
- **Data Preprocessing:**
  - Cleans missing values and converts date column to datetime format.
  - Creates new features such as temperature differences and moving averages.
  
- **Data Exploration:**
  - Summary of dataset using `df.info()`.
  - Checking for missing values and initial statistical analysis.
  
- **Model Selection & Training:**
  - Implements multiple regression models including Random Forest, Gradient Boosting, and Linear Regression.
  - Uses R2 score to evaluate model performance.
  
- **Hyperparameter Tuning:**
  - Uses Randomized Search Cross-Validation to optimize hyperparameters.
  
- **Visualization:**
  - Feature importance analysis for Random Forest model.
  - Comparison of model performances.
  - Actual vs Predicted temperature values.
  - Error distribution and feature correlation heatmap.

## Dependencies
Ensure you have the following Python libraries installed before running the script:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn fbprophet
```

## Usage
1. Place the `weather_data.csv` file in the project directory.
2. Run the script in a Python environment.
3. View the generated visualizations and model evaluation results.

## Results
The analysis provides insights into:
- The most significant features influencing temperature prediction.
- The best-performing model for weather forecasting.
- The effectiveness of time series forecasting using the Prophet model.


```

## Conclusion
This project showcases the end-to-end process of forecasting temperature using machine learning models. It evaluates different models, highlights the most significant features, and demonstrates the effectiveness of hyperparameter tuning for performance improvement.


# Airline Fare Prediction Using Machine Learning Algorithms

## Overview

This repository contains a project that aims to predict airline ticket prices using various machine learning algorithms. The project analyzes multiple factors influencing flight costs and leverages historical flight data to develop predictive models.

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Machine Learning Models](#machine-learning-models)
- [Result Analysis](#result-analysis)
- [Conclusion and Future Scope](#conclusion-and-future-scope)
- [Usage](#usage)
- [License](#license)

## Introduction

Airline ticket prices are influenced by numerous factors, including flight schedules, destinations, and seasonal trends. This project aims to provide insights into these factors through data analysis and machine learning, ultimately helping consumers make informed purchasing decisions.

## Data Collection

The project utilizes three datasets collected from different sources:

1. **MakeMyTrip Dataset**: Contains 10,000 records with features such as airline, source, destination, route, number of stops, price, and duration.
2. **EaseMyTrip Dataset**: A larger dataset with around 30,000 records focusing on similar features.
3. **New Zealand Airlines Dataset**: Data collected from two airlines over 90 days, including fare rates and flight details.

### Data Preprocessing

Data cleaning involved removing null values, encoding categorical data, and handling outliers to ensure the datasets were suitable for analysis.

## Machine Learning Models

The following machine learning algorithms were implemented using Python's Scikit-learn library:

- **KNN Regression**
- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Decision Tree Regression**
- **Stacking Regression**
- **Random Forest Regression**

Each model's performance was evaluated using metrics such as R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Result Analysis

The Random Forest Regressor outperformed other models with an average R² score of approximately 0.8 across all datasets. Detailed evaluation metrics are available in the repository.

### Example Code Snippet

```python
def predict(model):
    trained_model = model.fit(x_train, y_train)
    print("Training Score : {}".format(trained_model.score(x_train, y_train)))
    y_prediction = trained_model.predict(x_test)
    print("Predictions are : {}".format(y_prediction))
    print("Testing score : {}".format(trained_model.score(x_test, y_prediction)))
```

## Conclusion and Future Scope

This project demonstrates the potential of machine learning in predicting airline fare prices. Future work may include incorporating additional data sources or refining models for improved accuracy.

## Usage

To run this project:

1. Clone the repository.
2. Install required packages using `pip install -r requirements.txt`.
3. Execute the main script to train models and view results.

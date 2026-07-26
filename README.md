# 🏠 House Price Prediction using Linear Regression

Machine Learning Project — House Price Prediction

A Machine Learning model that predicts house prices using location, housing characteristics, income, population, rooms, bedrooms, and ocean proximity. The project includes data preprocessing, feature engineering, Linear Regression model training, evaluation, and a Flask-based web application.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

House prices depend on multiple factors such as location, number of rooms, number of bedrooms, population, household size, median income, and proximity to the ocean.

The objective of this project is to develop a machine learning model that can estimate the median house value from these housing characteristics.

---

## 🎯 Business Objective

Predict house prices from housing attributes to support:

- Preliminary property price estimation
- Real-estate data analysis
- Understanding factors affecting house prices
- Data-driven property valuation
- Machine Learning model experimentation

---

## 🗂️ Dataset

The project uses the **California Housing Dataset**.

| Property | Details |
|---|---|
| Dataset | `housing.csv` |
| Rows | 20,640 |
| Columns | 10 |
| Target | `median_house_value` |
| Missing Values | 207 in `total_bedrooms` |
| Duplicate Rows | 0 |

### Dataset Features

| Feature | Description |
|---|---|
| `longitude` | Longitude of the location |
| `latitude` | Latitude of the location |
| `housing_median_age` | Median age of houses |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Population of the area |
| `households` | Number of households |
| `median_income` | Median income of the area |
| `ocean_proximity` | Proximity to the ocean |
| `median_house_value` | Target house price |

---

## 🔄 Project Workflow

```text
Data Collection
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Duplicate Check
      ↓
Feature & Target Selection
      ↓
Categorical Feature Encoding
      ↓
Train/Test Split
      ↓
Linear Regression
      ↓
Prediction
      ↓
Model Evaluation
      ↓
Model Saving
      ↓
Flask Web Application
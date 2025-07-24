# Medical Cost Personal Insurance Prediction

This repository contains a Jupyter Notebook (`linear_regression.ipynb`) that explores and models medical insurance costs based on various personal attributes. The project utilizes linear regression to predict charges, with a focus on data exploration, visualization, and feature engineering.

## Table of Contents

- [Project Overview]
- [Dataset]
- [Analysis and Visualization]
- [Modeling]
- [Results]


## Project Overview

The goal of this project is to predict individual medical insurance charges using a linear regression model. The notebook covers:

- **Data Loading and Initial Inspection**: Importing necessary libraries, loading the dataset, and performing initial checks for data cleanliness and format.
- **Exploratory Data Analysis (EDA)**: Visualizing the distribution of key features (age, BMI, charges) and exploring relationships between variables, particularly how different factors influence insurance costs.
- **Feature Engineering**: Converting categorical features into numerical representations suitable for machine learning models.
- **Model Training and Evaluation**: Building and training a linear regression model, and evaluating its performance using metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
- **Optimization and Improvement**: Discussing strategies to enhance model accuracy.

## Dataset

The dataset used in this project is sourced from `INSURANCE_DATA_URL` (defined in `config.py`, which is assumed to be present). It contains the following columns:

- **age**: Age of the primary beneficiary
- **sex**: Gender of the primary beneficiary (male/female)
- **bmi**: Body mass index, providing an understanding of body weight relative to height
- **children**: Number of children covered by health insurance / number of dependents
- **smoker**: Whether the beneficiary smokes (yes/no)
- **region**: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest)
- **charges**: Individual medical costs billed by health insurance

### Initial Data Insights

- The dataset contains 1338 entries and 7 columns
- No missing values were found
- Columns `age`, `bmi`, `children`, and `charges` are numerical
- Columns `sex`, `smoker`, and `region` are categorical
- One duplicate entry was identified and removed, resulting in 1337 unique records

## Analysis and Visualization

The EDA section uses `matplotlib.pyplot`, `seaborn`, and `plotly.express` to visualize data distributions and relationships. Key observations include:

- **Age Distribution**: The distribution of age shows a higher concentration of customers in the 18-19 age group, suggesting potential demographic trends or specific insurance offerings for this age bracket.
- **BMI Distribution**: The BMI distribution is roughly normal, centered around 30, with most customers falling into overweight or obese categories.
- **Charges Distribution**: The distribution of charges is heavily skewed, resembling a "power law graph," where a small number of customers incur significantly higher costs.
- **Impact of Smoking**: Smoking status is a major predictor of insurance charges, with smokers generally facing much higher costs. The data suggests that obese smokers tend to pay significantly more than non-obese smokers.
- **Gender and Region Impact**: Visualizations also explore the impact of gender and region on insurance charges.

## Modeling

A linear regression model is used to predict medical charges.

### Initial Model (Numerical Features Only)

The first iteration of the model uses only numerical features (`age`, `bmi`, `children`).

**Equation:**
```
charges = w₁ × age + w₂ × bmi + w₃ × children + b
```

**Performance:**
- RMSE: ~11,358
- MAE: ~9,019

The initial model's performance was deemed inadequate due to high error rates.

### Improved Model (Including Categorical Features)

To improve accuracy, categorical features (`smoker`, `sex`, `region`) are converted to numerical representations using:

- **Binary Encoding**: For binary categories like smoker (yes/no converted to 1/0)
- **One-Hot Encoding**: For multi-category features like region

The model is then retrained with these engineered features.

**Equation (with smoker as an example):**
```
charges = w₁ × age + w₂ × bmi + w₃ × children + w₄ × smoker + b
```

**Performance after adding smoker:**
- RMSE: ~6,058
- MAE: ~4,181

Incorporating the smoker column significantly improved the model's performance, reducing the error by nearly half. Further improvements are expected with the inclusion of sex and one-hot encoded region features.

## Results

The analysis highlights that smoking status is the most influential factor in predicting medical insurance charges. While age and BMI also play a role, their impact is less pronounced, especially for non-smokers. The linear regression model, when properly fed with engineered categorical features, can provide a reasonable prediction of medical costs.


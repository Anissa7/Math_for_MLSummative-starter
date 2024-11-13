# Linear Regression Optimization with Gradient Descent

## Project Overview

This project demonstrates the implementation of Linear Regression to predict sales based on TV marketing expenses. Three different approaches are explored: 

1. **Linear Regression using NumPy**.
2. **Linear Regression using Scikit-Learn**.
3. **Linear Regression with Gradient Descent from scratch**.

Additionally, two other models, **Random Forest** and **Decision Trees**, are compared to linear regression in terms of performance based on RMSE (Root Mean Square Error).

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
    - Linear Regression using NumPy
    - Linear Regression using Scikit-Learn
    - Linear Regression with Gradient Descent
4. [Model Comparison](#model-comparison)
5. [Results](#results)
6. [Technologies Used](#technologies-used)
7. [Conclusion](#conclusion)

## Problem Definition

The task is to predict **Sales** based on the **TV marketing budget**. Given a dataset that contains two fields:
- **TV**: TV marketing expenses.
- **Sales**: Amount of sales generated.

## Dataset

The dataset used in this project is saved in `tvmarketing.csv`. It consists of 200 observations with two variables: `TV` marketing expenses and `Sales`.

## Methodology

### 1. Linear Regression using NumPy
We used the `np.polyfit` function to fit a polynomial of degree 1 to the dataset. The resulting slope and intercept allow us to create a simple prediction model using the equation:

\[ \text{Sales} = \text{TV} \times m + b \]

### 2. Linear Regression using Scikit-Learn
Scikit-Learn's `LinearRegression` model is utilized to fit the dataset. This approach offers the advantage of providing additional features such as automated prediction generation and easier handling of arrays.

Steps:
- Split the dataset into training and testing sets (80% training, 20% testing).
- Fit the linear regression model using training data.
- Predict the sales for the test data.
- Calculate RMSE to evaluate the performance.

### 3. Linear Regression with Gradient Descent
A custom implementation of linear regression using **Gradient Descent** was developed to minimize the sum of squared errors and optimize the model’s slope and intercept.

The cost function and its derivatives are computed, and the model is optimized using an iterative approach to minimize the error.

### 4. Random Forest and Decision Trees
Additionally, **Random Forest** and **Decision Trees** models are implemented to predict sales. The performance of each model is evaluated using RMSE and compared with the linear regression models.

## Model Comparison

The performance of each model was evaluated using **RMSE (Root Mean Square Error)** to measure the accuracy of the predictions. Here is a summary of the RMSE values for each model:

| Model             | RMSE         |
|-------------------|--------------|
| Linear Regression | 3.19         |
| Random Forest     | 3.01         |
| Decision Trees    | 3.44         |

## Results

The **Random Forest** model performed the best in terms of minimizing RMSE, followed by **Linear Regression** and **Decision Trees**.

## Technologies Used

- **Python**: Programming language.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation.
- **Scikit-Learn**: For machine learning algorithms.
- **Matplotlib**: For plotting and visualization.
- **Jupyter Notebook**: For developing and testing the code.

## Conclusion

This project highlights how different machine learning models can be used to solve a regression problem. We observed that even though **Linear Regression** is simpler, more complex models like **Random Forest** can often provide more accurate predictions.

# SGEMM Product Performance Prediction

## Overview

This project aims to predict the average run time of SGEMM (Single-Precision General Matrix Multiplication) operations using machine learning models. The goal is to optimize computational performance by analyzing various system and algorithmic parameters.

## Project Structure

- **`main.ipynb`**: Jupyter Notebook containing code for data preprocessing, EDA, model implementation, and evaluation.
- **`sgemm_product.csv`**: The dataset used for training and testing the models.

## Data Preprocessing

- **Data Cleaning**: Managed missing values and duplicates, and computed the average run time across multiple runs.
- **Feature Engineering**: Dropped unnecessary columns and scaled features for improved model performance.

## Exploratory Data Analysis (EDA)

- **Distribution Analysis**: Visualized the distribution of average run times.
- **Correlation Matrix**: Identified correlations between features.
- **Scatter Plots**: Explored relationships between key features and the target variable.

## Model Implementation

Three machine learning models were implemented:

1. **Linear Regression**: Provided a baseline performance.
2. **Decision Tree Regressor**: Captured non-linear relationships.
3. **Random Forest Regressor**: Delivered the best performance with the lowest RMSE and highest R² score.

## Model Evaluation

- **Metrics**: RMSE, R², and MAE were used to evaluate model performance.
- **Cross-Validation**: K-Fold cross-validation was applied to ensure robustness.

## Results

The **Random Forest Regressor** achieved the best results, making it the most effective model for predicting SGEMM run times.

## Conclusion

This project successfully applied machine learning to predict and optimize matrix operation performance, with the Random Forest model showing the highest accuracy.

## Technologies Used

- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sgemm-performance-prediction.git

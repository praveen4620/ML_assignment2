a. Problem Statement

The objective of this project is to predict whether an individual earns more than $50K per year using demographic and employment-related attributes from census data.

b. Dataset Description

The Adult Income dataset is sourced from the UCI Machine Learning Repository. It contains 48,842 records with 14 features including age, education, occupation, marital status, capital gain/loss, and hours worked per week.
The target variable is binary, indicating whether an individual earns more than $50K annually.

c. Models Used and Evaluation Metrics

| ML Model            | Accuracy |    AUC    | Precision | Recall   |    F1     |    MCC   |
| ------------------- | -------- | -------   | --------- | -------  | --------- | -------- |
| Logistic Regression | 0.845992 | 0.904650  | 0.731588  | 0.598127 | 0.658160  | 0.564786 |
| Decision Tree       | 0.805970 | 0.747334  | 0.603927  | 0.631133 | 0.617230  | 0.487575 |
| KNN                 | 0.824876 | 0.852031  | 0.665660  | 0.589652 | 0.625355  | 0.513233 |
| Naive Bayes         | 0.644776 | 0.850683  | 0.404636  | 0.918822 | 0.561844  | 0.411974 |
| Random Forest       | 0.845550 | 0.900425  | 0.721785  | 0.613292 | 0.663130  | 0.566941 | 
| XGBoost             | 0.865782 | 0.924871  | 0.767708  | 0.657449 | 0.708313  | 0.624979 | 

d. Model Observations

| ML Model            | Observation                                 |
| ------------------- | ------------------------------------------- |
| Logistic Regression | Performs well with proper feature encoding  |
| Decision Tree       | Tends to overfit categorical splits         |
| KNN                 | Sensitive to high-dimensional encoded data  |
| Naive Bayes         | Fast but limited by independence assumption |
| Random Forest       | Strong generalization and robustness        |
| XGBoost             | Best overall performance on all metrics     |

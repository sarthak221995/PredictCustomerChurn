# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

- Supervised Machine Learning Model to Identify the potential Customers to Churn

## Files and data description
.
├── data                                    # Training Data 
├── images                                  # EDA and Trained Model Results
├── logs                                    # Pytest Logs
├── models                                  # Trained Model
├── churn_library.py                        # Library to find customers who are likely to churn
├── churn_script_logging_and_tests.py       # Unit tests for the churn_library.py functions.
├── LICENSE
└── README.md 

## Running Files

**Train Churn Model**
- STEP 1 - Import DataSet
import churn_library
df = churn_library.import_data("./data/bank_data.csv")
 
- STEP 2 - Perform EDA
churn_library.perform_eda(df)

- STEP 3- Feature Engineering
x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(df, 'Churn')

- STEP 4 - Train Model
churn_library.train_models(x_train, x_test, y_train, y_test)

** Run Test **

pytest churn_script_logging_and_tests.py

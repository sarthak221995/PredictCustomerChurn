# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

- Supervised Machine Learning Model to Identify the potential Customers to Churn

## Files and data description
<pre>
├── data                                    # Training Data 
├── images                                  # EDA and Trained Model Results
├── logs                                    # Pytest Logs
├── models                                  # Trained Model
├── churn_library.py                        # Library to find customers who are likely to churn
├── churn_script_logging_and_tests.py       # Unit tests for the churn_library.py functions.
├── LICENSE
└── README.md 
</pre>

## Running Files

**Train Model**

- STEP 1 - Import DataSet
<pre>
import churn_library
df = churn_library.import_data("./data/bank_data.csv")
 </pre>
- STEP 2 - Perform EDA
<pre>
churn_library.perform_eda(df)
 </pre>
- STEP 3- Feature Engineering
<pre>
x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(df, 'Churn')
 </pre>
- STEP 4 - Train Model
<pre>
churn_library.train_models(x_train, x_test, y_train, y_test)
 </pre>
**Run Test**
<pre>
pytest churn_script_logging_and_tests.py
 </pre>

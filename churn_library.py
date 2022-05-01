# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:43:42 2022

@author: sarthak dargan sarthak221995@gmail.com
"""
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    print(df.columns)
    df['Churn'] = df.Attrition_Flag.apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    #df.Churn.plot.hist().get_figure()
    df.Churn.plot.hist()
    plt.savefig('images/eda/Churn.png')
    plt.close()
    
    plt.figure(figsize=(20, 10))
    
    df.Customer_Age.plot.hist().get_figure().savefig('images/eda/Customer_Age.png')
    df.Marital_Status.value_counts('normalize').plot(
        kind='bar').figure.savefig('images/eda/Marital_Status.png')
    sns.histplot(
        df['Total_Trans_Ct'],
        stat='density',
        kde=True).figure.savefig('images/eda/Total_Trans_Ct.png')
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r',
                linewidths=2).figure.savefig('images/eda/df_corr.png')
   
    return None


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # gender encoded column
    y = response
    
    for category in category_lst:
            
        temp_lst = []
        temp_groups = df.groupby(category).mean()[y]
    
        for val in df[category]:
            temp_lst.append(temp_groups.loc[val])
    
        df['{0}_Churn'.format(category)] = temp_lst
    
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
    
    df = encoder_helper(df, cat_columns, response)
    y = df[response]
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    df = df[keep_cols]
    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.clf()
    plt.rc("figure", figsize=(5, 5))
    plt.text(0.01,
             1.50,
             str(classification_report(y_test, y_test_preds_rf)),
             {"fontsize": 10},
             fontproperties="monospace")
    
    plt.text(0.01,
             2.00,
             str(classification_report(y_train, y_train_preds_rf)),
             {"fontsize": 10},
             fontproperties="monospace")
    
    plt.text(0.01,
             3.00,
             str(classification_report(y_test, y_test_preds_lr)),
             {"fontsize": 10},
             fontproperties="monospace")
    
    plt.text(0.01,
             4.00,
             str(classification_report(y_train, y_train_preds_lr)),
             {"fontsize": 10},
             fontproperties="monospace")
    
    plt.savefig("images/results/rf_test_cr_results.jpg")
    plt.close()
       
    return None


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    

    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(20,8))
    
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    
    plt.savefig(output_pth)
    
    return None
    

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    feature_importance_plot(cv_rfc.best_estimator_, X_train, "images/results/feature_importance.jpg")

    return None


if __name__ == "__main__":

    # STEP 1 - DONE
    df = import_data("./data/bank_data.csv")

    # STEP 2 - DONE
    perform_eda(df)
    
    # STEP 3- DONE
    X_train, X_test, y_train, y_test = perform_feature_engineering(df,'Churn')

    # STEP 4 -
    train_models(X_train, X_test, y_train, y_test)
    
    #model = joblib.load('./models/rfc_model.pkl')
    
#
#    print(df.shape)

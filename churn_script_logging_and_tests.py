# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:47:22 2022

@author: sarthak dargan sarthak221995@gmail.com
"""

import os
import pytest
import glob
import logging
import churn_library

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name='df_original')
def df_original_():
    """
    Original dataframe fixture - returns the original dataframe
    """
    try:
        df_original = churn_library.import_data("./data/bank_data.csv")
        logging.info("Original dataframe fixture: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Original dataframe fixture: The file wasn't found")
        raise err
    return df_original


@pytest.fixture(name='df_encoded')
def df_encoded_(df_original):
    """
    Encoded dataframe fixture - returns the dataframe after Encoding of Categorical Features
    """
    try:
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']

        response = 'Churn'

        df_encoded = churn_library.encoder_helper(
            df_original, cat_columns, response)

        logging.info("Encoded dataframe fixture: SUCCESS")
    except KeyError() as err:
        logging.error(
            "Encoded dataframe fixture: Required Features Not Present")
        raise err
    return df_encoded


@pytest.fixture(name='df_processed')
def df_processed_(df_original):
    """
    Processed dataframe fixture - returns the Train Test Split Processed Data
    """
    try:
        x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(
            df_original, 'Churn')
        logging.info("Processed dataframe fixture: SUCCESS")
    except KeyError() as err:
        logging.error(
            "Processed dataframe fixture: Feature Engineering Failed")
        raise err
    return x_train, x_test, y_train, y_test


def test_import(df_original):
    '''
    test data import - test if orginal dataset loaded
    '''
    try:
        assert not df_original.empty
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df_original.shape[0] > 0
        assert df_original.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_original):
    '''
    test perform eda function
    '''
    try:
        churn_library.perform_eda(df_original)
        assert os.path.exists('./images/eda/Churn.png')
        assert os.path.exists('./images/eda/Customer_Age.png')
        assert os.path.exists('./images/eda/df_corr.png')
        assert os.path.exists('./images/eda/Marital_Status.png')
        assert os.path.exists('./images/eda/Total_Trans_Ct.png')
    except AssertionError as err:
        logging.error("Testing eda: Images Not Generated")
        raise err


def test_encoder_helper(df_encoded):
    '''
        test encoder helper
    '''
    try:
        assert not df_encoded.empty
    except AssertionError as err:
        logging.error("Testing Encoder Helper: Required Features Not Present")
        raise err


def test_perform_feature_engineering(df_processed):
    '''
        test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = df_processed
        assert x_train.shape[0] == len(y_train)
        assert x_test.shape[0] == len(y_test)
    except AssertionError as err:
        logging.error(
            "Testing Feature Engineering: Feature Engineering Failed ")
        raise err


def test_train_models(df_processed):
    '''
        test train_models
    '''
    try:
        x_train, x_test, y_train, y_test = df_processed
        churn_library.train_models(x_train, x_test, y_train, y_test)
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./models/rfc_model.pkl')
    except AssertionError as err:
        logging.error("Testing Models: Trained Models Not Available ")
        raise err


# if __name__ == "__main__":
#     for directory in ["./images/eda", "./images/results", "./models"]:
#         files = glob.glob("%s/*" % directory)
#         for file in files:
#             os.remove(file)
#     pytest.main(["-s"])

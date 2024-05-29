from lib_ml import preprocess_input
import os
from joblib import load
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture()
def char_index():
    char_index = load('data/interim/char_index.joblib')
    yield char_index


@pytest.fixture()
def x_data():
    x_test = load('data/interim/x_data.joblib')
    yield x_test


@pytest.fixture()
def y_data():
    y_test = load('data/interim/y_data.joblib')
    yield y_test


# Features and data test
def test_preprocessing():
    """ Tests whether links are processed correctly. """
    char_index, tokenized_x = preprocess_input("https://www.tudelft.nl/")
    assert isinstance(char_index, dict)
    assert tokenized_x.shape[1] == 200

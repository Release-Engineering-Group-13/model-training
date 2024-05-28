#from sklearn.externals import joblib
import os
from joblib import load
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.model.model_train import model_train
from src.model.model_predict import model_predict
from lib_ml import preprocess_dataset, preprocess_input

char_index = load('data/interim/char_index.joblib')

@pytest.fixture()
def trained_model():
    trained_model = load('data/interim/model.joblib')
    yield trained_model

@pytest.fixture()
def x_data():
    x_test = load('data/interim/x_data.joblib')
    yield x_test

@pytest.fixture()
def y_data():
    y_test = load('data/interim/y_data.joblib')
    yield y_test

@pytest.fixture()
def predictions(trained_model, x_data):
    y_pred = trained_model.predict(x_data[2], batch_size=1000)
    yield y_pred

#def test_nondet_robustness(x_data, y_data, trained_model):
    #Tests whether the model is robust to different random seeds. 
    """
    original_acc = trained_model.evaluate(x_data[2], y_data[2])[1]
    print("original acc: ", original_acc)
    for seed in [1,2]:
        model_variant, _ = model_train(x_data[0], y_data[0], x_data[1], y_data[1], char_index, seed)
        print("Done training variant")
        variant_acc = model_variant.evaluate(x_data[2], y_data[2])[1]
        print("variant acc: ", variant_acc)
        print(abs(original_acc - variant_acc))
        assert abs(original_acc - variant_acc) <= 0.03"""


# Code to test preprocessing (features and data test)
# Test on code slice (model development)
# Code to test if predictions are between 0 and 1 (or phishing and legitimate, technically features and data i think??), and if they're not NaN (monitoring tests)
def test_prediction_values_validity(x_data, trained_model, predictions):
    #y_pred = trained_model.predict(x_data[2], batch_size=1000)
    assert np.nan not in predictions and np.inf not in predictions

def test_prediction_values_range(x_data, trained_model, predictions):
    #y_pred = trained_model.predict(x_data[2], batch_size=1000)
    assert np.all((predictions >= 0)|(predictions <= 1 ))

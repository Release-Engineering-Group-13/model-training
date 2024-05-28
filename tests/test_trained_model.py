import os
from joblib import load
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.model.model_train import model_train
from lib_ml import preprocess_dataset, preprocess_input

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

@pytest.fixture()
def trained_model():
    trained_model = load('data/interim/model.joblib')
    yield trained_model

@pytest.fixture()
def predictions(trained_model, x_data):
    y_pred = trained_model.predict(x_data[2], batch_size=1000)
    yield y_pred

# Model development test
def test_nondet_robustness(x_data, y_data, trained_model, char_index):
    """ Tests whether the model is robust to different random seeds. """
    original_acc = trained_model.evaluate(x_data[2], y_data[2])[1]
    print("original acc: ", original_acc)
    for seed in [1,2]:
        model_variant, _ = model_train(x_data[0], y_data[0], x_data[1], y_data[1], char_index, seed)
        print("Done training variant")
        variant_acc = model_variant.evaluate(x_data[2], y_data[2])[1]
        print("variant acc: ", variant_acc)
        print(abs(original_acc - variant_acc))
        assert abs(original_acc - variant_acc) <= 0.03


# Features and data test
def test_preprocessing():
    """ Tests whether links are processed correctly. """
    char_index, tokenized_x = preprocess_input("https://www.tudelft.nl/en/student/administration/termination-of-enrolment")
    assert isinstance(char_index, dict)
    assert tokenized_x.shape[1] == 200  

# TODO: Test on data slice (model development)

# Infrastructure test
def test_loss_decrease(x_data, y_data, char_index):
    """ Tests whether training leads to reduced loss. """
    sub_size = 1000
    model, hist = model_train(x_data[0][:sub_size], y_data[0][:sub_size], x_data[1][:sub_size], y_data[1][:sub_size], char_index)
    hist = model.fit(x_data[0][:sub_size], y_data[0][:sub_size], 
                             batch_size=100, epochs=3, shuffle=True, 
                             validation_data=(x_data[1][:sub_size], y_data[1][:sub_size]))
    loss_first = hist.history['loss'][0]
    loss_final = hist.history['loss'][-1]
    assert loss_first > loss_final     

# Monitoring test
def test_prediction_values_validity(predictions):
    """ Tests if predictions are between 0 and 1, and if they're not NaN or infinity. """
    assert np.nan not in predictions and np.inf not in predictions
    assert np.all((predictions >= 0)|(predictions <= 1 ))

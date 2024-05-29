import numpy as np
from src.model.model_train import model_train
import os
from joblib import load
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


SUB_SIZE = 1000  # size of subset of data


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
def test_nondet_robustness(x_data, y_data, char_index):
    """ Tests whether the model is robust to different random seeds. """
    model_1, _ = model_train(x_data[0][:SUB_SIZE], y_data[0][:SUB_SIZE],
                             x_data[1][:SUB_SIZE], y_data[1][:SUB_SIZE],
                             char_index)
    acc_1 = model_1.evaluate(x_data[2][:SUB_SIZE], y_data[2][:SUB_SIZE])[1]
    model_2, _ = model_train(x_data[0][:SUB_SIZE], y_data[0][:SUB_SIZE],
                             x_data[1][:SUB_SIZE], y_data[1][:SUB_SIZE],
                             char_index, seed=45)
    acc_2 = model_2.evaluate(x_data[2][:SUB_SIZE], y_data[2][:SUB_SIZE])[1]
    assert abs(acc_1 - acc_2) <= 0.03


# TODO: Test on data slice (model development)


# Infrastructure test
def test_loss_decrease(x_data, y_data, char_index):
    """ Tests whether training leads to reduced loss. """
    model, hist = model_train(x_data[0][:SUB_SIZE], y_data[0][:SUB_SIZE],
                              x_data[1][:SUB_SIZE], y_data[1][:SUB_SIZE],
                              char_index)
    hist = model.fit(x_data[0][:SUB_SIZE], y_data[0][:SUB_SIZE],
                     batch_size=100, epochs=3, shuffle=True,
                     validation_data=(x_data[1][:SUB_SIZE], y_data[1][:SUB_SIZE]))
    loss_first = hist.history['loss'][0]
    loss_final = hist.history['loss'][-1]
    assert loss_first > loss_final


# Monitoring test
def test_prediction_values_validity(predictions):
    """ Tests if predictions are between 0 and 1, and if they're not NaN or infinity. """
    assert np.nan not in predictions and np.inf not in predictions
    assert np.all((predictions >= 0) | (predictions <= 1))

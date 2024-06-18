# flake8: noqa
import numpy as np
from src.model.model_train import model_train
from fixtures import char_index, x_data, y_data, trained_model, processed_phishing_links, preprocessed_phising_links
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# data type test
def test_char_index(char_index):
    """ Tests if the character index is a dictionary. """
    assert isinstance(char_index, dict)


def test_x_data(x_data):
    """ Tests if the data is a list of arrays. """
    assert isinstance(x_data, list)
    assert all(isinstance(x, np.ndarray) for x in x_data)


def test_y_data(y_data):
    """ Tests if the data is a list of arrays. """
    assert isinstance(y_data, list)
    assert all(isinstance(y, np.ndarray) for y in y_data)


def test_trained_model(trained_model):
    """ Tests if the model is a Keras model. """
    assert hasattr(trained_model, 'evaluate')
    assert hasattr(trained_model, 'predict')


def test_phishing_links(processed_phishing_links):
    """ Tests if the phishing links are a list. """
    assert isinstance(processed_phishing_links, list)


def test_phishing_links_content(preprocessed_phising_links):
    """ Tests if the phishing links are strings. """
    assert all(isinstance(link, str) for link in preprocessed_phising_links)


def test_phishing_links_length(processed_phishing_links):
    """ Tests if the phishing links are not empty. """
    assert all(len(link) > 0 for link in processed_phishing_links)


# Model development test
def test_nondet_robustness(x_data, y_data, char_index, trained_model):
    """ Tests whether the model is robust to different random seeds. """
    acc_1 = trained_model.evaluate(x_data[2], y_data[2])[1]
    model_2, _ = model_train(x_data[0], y_data[0],
                             x_data[1], y_data[1],
                             char_index, seed=2952024,
                             batch_train=50, batch_test=50)
    acc_2 = model_2.evaluate(x_data[2], y_data[2])[1]
    assert abs(acc_1 - acc_2) <= 0.025


# Test on data slice (model development)
def test_phishing_links(processed_phishing_links, trained_model):
    """ Tests whether prediction performance on phishing links is adequate. """
    y = np.array([0] * len(processed_phishing_links))
    acc_phishing = trained_model.evaluate(processed_phishing_links, y)[1]
    assert acc_phishing > 0.80


# Infrastructure test
def test_loss_decrease(x_data, y_data, trained_model):
    """ Tests whether training leads to reduced loss. """
    hist = trained_model.fit(x_data[0], y_data[0],
                             batch_size=50, epochs=3, shuffle=True,
                             validation_data=(x_data[1], y_data[1]))
    loss_first = hist.history['loss'][0]
    loss_final = hist.history['loss'][-1]
    assert loss_first > loss_final


# Monitoring test
def test_prediction_values_validity(x_data, trained_model):
    """ Tests if predictions are between 0 and 1, and if they're not NaN or infinity. """
    predictions = trained_model.predict(x_data[2], batch_size=1000)
    assert np.nan not in predictions and np.inf not in predictions
    assert np.all((predictions >= 0) | (predictions <= 1))


# Monitoring test
def test_prediction_values_consistency(x_data, trained_model):
    """ Tests if predictions are consistent across different batch sizes. """
    predictions_1 = trained_model.predict(x_data[2], batch_size=1000)
    predictions_2 = trained_model.predict(x_data[2], batch_size=1)
    assert np.allclose(predictions_1, predictions_2, atol=1e-2)


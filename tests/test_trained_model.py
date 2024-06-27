'''Test trained model'''

# flake8: noqa
import os
import sys
import numpy as np
from src.model.model_train import model_train
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fixtures import char_index, x_data, y_data, trained_model, phishing_links


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
def test_phishing_links(phishing_links, trained_model):
    """ Tests whether prediction performance on phishing links is adequate. """
    y = np.array([0] * len(phishing_links)) 
    acc_phishing = trained_model.evaluate(phishing_links, y)[1]
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

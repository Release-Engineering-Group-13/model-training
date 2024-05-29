from src.preprocessing.preprocessing import main
from src.model.model_train import model_train
import os
from joblib import load
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

OUT_FOLDER = "tests/test_data/"

main("tests/test_data/", "tests/test_data/", "train.txt", "test.txt")


@pytest.fixture()
def char_index():
    char_index = load(f'{OUT_FOLDER}/char_index.joblib')
    yield char_index


@pytest.fixture()
def x_data():
    x_test = load(f'{OUT_FOLDER}/x_data.joblib')
    yield x_test


@pytest.fixture()
def y_data():
    y_test = load(f'{OUT_FOLDER}/y_data.joblib')
    yield y_test


@pytest.fixture()
def trained_model(x_data, y_data, char_index):
    trained_model, _ = model_train(x_data[0], y_data[0], x_data[1], y_data[1], char_index, batch_train=50, batch_test=50)
    yield trained_model


@pytest.fixture()
def predictions(trained_model, x_data):
    y_pred = trained_model.predict(x_data[2], batch_size=1000)
    yield y_pred

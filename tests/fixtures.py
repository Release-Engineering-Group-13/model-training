'''Text fixtures'''

import sys
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import pytest
from src.preprocessing.preprocessing import main
from src.model.model_train import model_train

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

OUT_FOLDER = "tests/test_data/"

main("tests/test_data/", "tests/test_data/", "train.txt", "test.txt")


@pytest.fixture()
def char_index():
    """Load char index"""
    char_index_result = load(f'{OUT_FOLDER}/char_index.joblib')
    yield char_index_result


@pytest.fixture()
def x_data():
    """Load x test data"""
    x_data = load(f'{OUT_FOLDER}/x_data.joblib')
    yield x_data


@pytest.fixture()
def y_data():
    """Load y test data"""
    y_test = load(f'{OUT_FOLDER}/y_data.joblib')
    yield y_test


@pytest.fixture()
def trained_model(x_data, y_data, char_index):
    """Train model"""
    trained_model_result, _ = model_train(x_data[0], y_data[0],
                                          x_data[1], y_data[1],
                                          char_index, batch_train=50, batch_test=50)
    yield trained_model_result


@pytest.fixture()
def phishing_links():
    """ Extracts phishing links from the test file. """
    dataset_folder = "tests/test_data/"
    file = "test.txt"
    phishing_links_result = []

    with open(dataset_folder + file, "r", encoding="utf-8") as lines:
        for line in lines:
            label, link = line.strip().split("\t")
            if label == "phishing":
                phishing_links_result.append(link)

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(phishing_links_result)  # Fit on phishing links only for this fixture
    sequence_length = 200
    processed_links = pad_sequences(tokenizer.texts_to_sequences(phishing_links_result),
                                    maxlen=sequence_length)

    yield np.array(processed_links)

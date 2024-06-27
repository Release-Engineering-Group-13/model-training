'''Test preprocessing'''

# flake8: noqa
import os
import sys
from lib_ml import preprocess_input
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Features and data test
def test_preprocessing():
    """ Tests whether links are processed correctly. """
    char_index, tokenized_x = preprocess_input("https://www.tudelft.nl/")
    assert isinstance(char_index, dict)
    assert tokenized_x.shape[1] == 200

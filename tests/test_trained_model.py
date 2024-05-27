#from sklearn.externals import joblib
import os
from joblib import load
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.model_train import model_train
from src.model.model_predict import model_predict

#input_folder = "data/interim"
#if not os.path.exists(input_folder):
#    raise FileNotFoundError(f"Input folder '{input_folder}' is empty")
char_index = load('../data/interim/char_index.joblib')
#x = load('data/interim/x_data.joblib')
#y = load('data/interim/y_data.joblib')

@pytest.fixture()
def trained_model():
    #input_folder = "data/interim"
    #if not os.path.exists(input_folder):
    #    raise FileNotFoundError(f"Input folder '{input_folder}' is empty")
    trained_model = load('../data/interim/model.joblib')
    yield trained_model

@pytest.fixture()
def x_test_data():
    x_test = load('../data/interim/x_data.joblib')
    yield x_test

@pytest.fixture()
def y_test_data():
    y_test = load('../data/interim/y_data.joblib')
    yield y_test

def test_nondet_robustness(x_test_data, y_test_data, trained_model):
    original_acc = trained_model.evaluate(x_test_data[2], y_test_data[2])[1]
    #original_acc = model_predict(x_test_data[2], y_test_data[2], trained_model)[2] 
    print("original acc: ", original_acc)
    for seed in [1,2]:
        model_variant, _ = model_train(x_test_data[0], y_test_data[0], x_test_data[1], y_test_data[1], char_index, seed)
        print("Done training variant")
        variant_acc = model_variant.evaluate(x_test_data[2], y_test_data[2])[1]
        #variant_acc = model_predict(x_test_data, y_test_data, model_variant)[2] 
        print("variant acc: ", variant_acc)
        print(abs(original_acc - variant_acc))
        assert abs(original_acc - variant_acc) <= 0.03

# Code to test preprocessing (features and data test)
# Test on code slice (model development)
# Code to test if predictions are between 0 and 1 (or phishing and legitimate, technically features and data i think??), and if they're not NaN (monitoring tests)
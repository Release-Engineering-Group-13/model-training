
<!-- badge-start -->
![Pytest](https://img.shields.io/badge/pylint-passing-brightgreen?logo=pylint)
![Pytest](https://img.shields.io/badge/pytest-passing-brightgreen?logo=pytest)
![Pytest](https://img.shields.io/badge/flake8-passing-brightgreen?logo=flake8)
<!-- badge-end -->


# Model training
Responsible for the training of the machine learning model


## Setup
To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Release-Engineering-Group-13/CS4295_FinalProject.git
    ```

2. Install Poetry
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Install the required dependencies using Poetry:
    ```bash
    poetry install
    ```

4. Enter the poetry environment to run DVC commands:
    ```bash
    poetry shell


Windows users might run into errors when trying to use Poetry for this project (see [open issue](https://github.com/tensorflow/io/issues/1789)). In that case, use a venv virtual environment:

2.  Create a virtual environment
    ```bash
    python -m venv yourenv
    ```
    
3. Install requirements in virtual environment
    ```bash
    source yourenv/bin/activate
    pip install -r requirements.txt
    ```

4. Authentication

    In order to download the dataset, you must first authenticate using an kaggle API token. Go to the 'Account' tab of your user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.

    If you are using the Kaggle CLI tool, the tool will look for this token at ~/.kaggle/kaggle.json on Linux, OSX, and other UNIX-based operating systems, and at C:\Users\\\<Windows-username\>\\.kaggle\kaggle.json on Windows. If the token is not there, an error will be raised. Hence, once you’ve downloaded the token, you should move it from your Downloads folder to this folder.

5. To exit the virtual environment (Poetry or venv), type
    ```bash
    exit
    ```

## Usage
Run the code using DVC:
```bash
dvc repro
```

## Metrics
To run an experiment with DVC, use:
```bash
dvc exp run
```

To see the difference between experiments, run:
```bash
dvc metrics diff
```

Show different experiments with:
```bash
dvc exp show
```

## Running linters
1. Run pylint:
   ```bash
   pylint path\to\file.py
   ```
2. Run flake8:
   ```bash
   flake8 --max-line-length=100 path\to\file.py
   ```

## Running pytest
It is important that pytest is executed in the model-training directory, otherwise it'll fail to find the data files. The data files in test_data each consist of 1000 links and their ground truth labels. 
1. Run pytest:
   ```bash
   pytest
   ```

## Authors
- Esha Dutta
- Vanessa Timmer
- Maarten van Bijsterveldt 
- Nick Dubbeldam

## Tests
<!-- results-start -->

### Pytest Results

- **Tests:** 1
- **Errors:** 0
- **Failures:** 0
- **Skipped:** 0


### Pylint Results

- **Tests:** 5
- **Errors:** 0
- **Failures:** 0
- **Skipped:** 0


### Flake8 Results

- **Tests:** 1
- **Errors:** 0
- **Failures:** 0
- **Skipped:** 0

<!-- results-end -->

## License

[MIT](https://choosealicense.com/licenses/mit/)


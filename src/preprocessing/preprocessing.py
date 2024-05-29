"""This module prepocesses the data in order to make it usable by a model."""

from lib_ml import preprocess_dataset


# main function
def main(dataset_folder="data/raw/DL Dataset/", output="data/interim",
         train="train.txt", test="test.txt"):
    """Preprocesses the data and stores it in a folder."""

    preprocess_dataset(dataset_folder, train_file=train, test_file=test, output_folder=output)


if __name__ == "__main__":
    main()

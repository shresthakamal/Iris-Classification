import os

import pandas as pd
from sklearn.model_selection import train_test_split

from irisclassification.config import config
from irisclassification.utils.seralizer import save_object


def build_features(dataset_path, split_ratio=0.3):

    iris = pd.read_csv(dataset_path)

    """DATA PRE-PROCESSING

    No missing data
    No null values
    No Encoding of categorical values needed
    No Standaridastaion needed
    No ill formated values

    """

    train, test = train_test_split(iris, test_size=config.TEST_SIZE)

    train_x = train.loc[:, train.columns != "variety"]

    train_y = train.loc[:, train.columns == "variety"]

    test_x = test.loc[:, test.columns != "variety"]

    test_y = test.loc[:, test.columns == "variety"]

    save_object(
        filepath=os.path.join(config.DATA_PATH, "processed"),
        filename="pre_processed_data",
        object_arr=[iris],
    )

    save_object(
        filepath=os.path.join(config.CHECKPOINTS_PATH),
        filename="train_data",
        object_arr=[train],
    )

    save_object(
        filepath=os.path.join(config.CHECKPOINTS_PATH),
        filename="test_data",
        object_arr=[test],
    )

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = build_features(
        dataset_path=os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME),
        split_ratio=config.TEST_SIZE,
    )

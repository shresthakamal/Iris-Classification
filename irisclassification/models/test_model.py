import os
import pickle

from irisclassification.config import config
from irisclassification.features.build_features import build_features


class TestModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.clf = TestModel.load_model(model_name)

    @staticmethod
    def load_model(model_name):
        if os.path.exists(
            os.path.join(config.CHECKPOINTS_PATH, "{}.pkl".format(model_name))
        ):
            with open(
                os.path.join(
                    config.CHECKPOINTS_PATH, "{}.pkl".format(model_name)
                ),
                "rb",
            ) as handle:
                clf = pickle.load(handle)
            return clf
        else:
            return None

    def predict(self, **kwargs):
        if self.clf:
            predictions = self.clf.predict(kwargs["test_x"])
            return predictions
        else:
            print("No saved model to predict !!")
            return None


if __name__ == "__main__":
    tester = TestModel("knn")

    x_train, y_train, x_test, y_test = build_features(
        dataset_path=os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME),
        split_ratio=config.TEST_SIZE,
    )
    predictions = tester.predict(test_x=x_test)

    print(predictions)

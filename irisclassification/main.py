import os

from MultiChoice import MultiChoice

from irisclassification.config import config
from irisclassification.config.model_params import parameters
from irisclassification.features.build_features import build_features
from irisclassification.models import test_model, train_model
from irisclassification.utils.model_accuracy import model_accuracy


def train_pipeline(model_name, x_train, y_train):

    model_params = parameters[model_name]

    trainer = train_model.TrainModel(model_name, **model_params)

    trainer.fit(train_x=x_train, train_y=y_train)

    return True


def test_pipeline(model_name, x_test):

    tester = test_model.TestModel(model_name)

    predictions = tester.predict(test_x=x_test)

    return predictions


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = build_features(
        dataset_path=os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME),
        split_ratio=config.TEST_SIZE,
    )

    # User's Model Choice
    user_selected_model = MultiChoice(
        "Select one of the following models:",
        options=("knn", "svm", "logisticregression", "decisiontree"),
    )().lower()

    train_pipeline(user_selected_model, x_train, y_train)

    predictions = test_pipeline(user_selected_model, x_test)

    print(model_accuracy(predictions, y_test))

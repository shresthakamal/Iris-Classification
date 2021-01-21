import logging
import os
import pickle

from irisclassification.config import config, model_params
from irisclassification.dispatcher import dispatcher
from irisclassification.utils.log import Log
from irisclassification.utils.seralizer import save_object

# from irisclassification.features.build_features import build_features


class TrainModel:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.model = dispatcher.MODELS[model_name](**kwargs)

    def fit(self, **kwargs):

        Log.init()
        logging.info("Training with model: {}".format(self.model_name))

        clf = self.model.fit(
            kwargs["train_x"], kwargs["train_y"].values.ravel()
        )

        save_object(
            filepath=os.path.join(config.CHECKPOINTS_PATH, "models"),
            filename="{}".format(self.model_name),
            object_arr=[clf],
        )
        return clf


if __name__ == "__main__":
    pass

    # model_params = {"n_neighbors": 4}

    # trainer = TrainModel("knn", **model_params)

    # x_train, y_train, x_test, y_test = build_features(
    #     dataset_path=os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME),
    #     split_ratio=config.TEST_SIZE,
    # )

    # print(trainer.fit(train_x=x_train, train_y=y_train))

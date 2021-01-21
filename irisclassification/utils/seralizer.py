import logging
import os
import pickle

from irisclassification.utils.log import Log


def save_object(filepath, filename, _object):

    Log.init()

    logging.info("Saving {}".format(filename))

    with open(os.path.join(filepath, f"{filename}.pkl"), "wb") as handle:
        pickle.dump(_object, handle)

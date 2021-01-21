import os
import unittest

from irisclassification.config import config
from irisclassification.data import make_dataset


class test_dataset(unittest.TestCase):
    def test_make_dataset(self):
        self.assertEqual(
            make_dataset.download_dataset(config.DATASET_URL),
            True,
            "[ERROR/test_make_dataset]: Test Failed !",
        )

    def test_dataset(self):
        self.assertEqual(
            os.path.exists(
                os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME)
            ),
            True,
            "[Error/test_dataset]: Test Failed !!",
        )


if __name__ == "__main__":
    unittest.main()

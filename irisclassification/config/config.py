import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATASET_NAME = "Iris.csv"

DATA_PATH = os.path.join(BASE_DIR, "data")

DATASET_URL = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"

CHECKPOINTS_PATH = os.path.join(BASE_DIR, "checkpoints")

LOG_FILE = os.path.join(CHECKPOINTS_PATH, "app.log")

TEST_SIZE = 0.2


if __name__ == "__main__":
    pass

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

MODELS = {
    "logisticregression": LogisticRegression,
    "svm": svm.SVC,
    "decisiontree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
}


if __name__ == "__main__":
    pass

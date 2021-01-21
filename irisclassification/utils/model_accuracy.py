from sklearn.metrics import accuracy_score


def model_accuracy(predictions, actual):
    return accuracy_score(predictions, actual)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import _pickle
import random
import gzip
import math


def sigmoid(x):
    return 1 / (1 + (math.e ** (-x)))


def supervised_error(target, output):
    target_label = target.argmax()
    output_label = output.argmax()
    if target_label != output_label:
        return 1
    return 0


def init_random(n, m):
    return 2 * np.random.rand(n, m) - 1


def load_data(path):
    with gzip.open(path, 'rb') as fd:
        train_set, valid_set, test_set = _pickle.load(fd, encoding='latin')
        return train_set, valid_set, test_set


def convert(dataset):
    c_set = []
    for i in range(len(dataset[0])):
        input_ = dataset[0][i]
        label_ = np.zeros((10, 1), dtype='float32')
        label_[dataset[1][i]] = 1
        c_set.append([np.array(input_).flatten().reshape((28 * 28, 1)), label_,
                      np.array(input_).flatten().reshape((1, 28 * 28))])
    return c_set


def get_batches(train_set, n):
    batch_len = len(train_set) // n
    batches = []
    for i in range(n):
        batch = []
        for j in range(i * batch_len, (i + 1) * batch_len):
            batch.append(train_set[j])
        batches.append(batch)
    return batches


def shuffle(train_set):
    return random.sample(train_set, len(train_set))


def apply_activation(output):
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = sigmoid(output[i, j])
    return output


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # plt.show()
    plt.savefig("confusion_matrix.png")


def accuracy_plot(accuracy_history):
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_history, label="Accuracy", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("accuracy.png")

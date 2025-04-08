from .utils import euclidean_distance
import matplotlib.pyplot as plt
import numpy as np


def knn_predict(X_train, y_train, X_test, k=5):
    y_pred = []
    for test_sample in X_test:
        distances = [euclidean_distance(test_sample, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common)
    return np.array(y_pred)


def plot_classified_points(X_train, X_test, y_train, y_test):
    y_train_symbols = ['o' if label == 0 else 's' for label in y_train]
    y_test_symbols = ['o' if label == 0 else 's' for label in y_test]

    plt.figure(figsize=(8, 6))
    for i, symbol in enumerate(y_train_symbols):
        plt.scatter(X_train[i, 0], X_train[i, 1], marker=symbol, color='blue', alpha=0.5,
                    label='Train' if i == 0 else "")
    for i, symbol in enumerate(y_test_symbols):
        plt.scatter(X_test[i, 0], X_test[i, 1], marker=symbol, color='red', alpha=0.5, label='Test' if i == 0 else "")
    plt.xlabel("IMC")
    plt.ylabel("Colesterol")
    plt.title("Clasificarea punctelor (Train vs Test)")
    plt.legend()
    plt.savefig("clasificarea_punctelor.png")


def plot_misclassified_points(X_test, y_test, y_pred):
    plt.figure(figsize=(8, 6))
    for i in range(len(X_test)):
        if y_pred[i] == y_test[i]:
            plt.scatter(X_test[i, 0], X_test[i, 1], color='green', marker='o', alpha=0.6,
                        label='Corect' if i == 0 else "")
        else:
            plt.scatter(X_test[i, 0], X_test[i, 1], color='red', marker='x', alpha=0.6,
                        label='Gresit' if i == 0 else "")
    plt.xlabel("IMC")
    plt.ylabel("Colesterol")
    plt.title("Clasificare corectă vs incorectă")
    plt.legend()
    plt.savefig("clasificare_corecta_vs_incorecta.png")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop("TYPE", axis=1)
    y = df["TYPE"]
    return X, y


def preprocess_data(X, y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return train_test_split(X, y_encoded, test_size=0.2, random_state=None), encoder


def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def plot_loss(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("loss_curve.png")


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


def class_distribution(y_train):
    train_label_counts = pd.Series(y_train).value_counts()
    print("Distribuția pe clase în antrenament:\n", train_label_counts)
    return train_label_counts


def plot_class_distribution(train_label_counts):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=train_label_counts.index, y=train_label_counts.values, hue=train_label_counts.index, palette="coolwarm", legend=False)
    plt.title("Distribuția claselor în setul de antrenament")
    plt.ylabel("Număr de exemple")
    plt.xlabel("Tip afecțiune")
    plt.tight_layout()
    # plt.show()
    plt.savefig("class_distribution.png")


def balance_and_split_data(X_train, y_train, train_label_counts):
    desired_class_size = max(train_label_counts)
    print(f"Dimensiunea dorită pentru fiecare clasă: {desired_class_size}")

    X_resampled = []
    y_resampled = []

    for label in train_label_counts.index:
        class_samples = X_train[y_train == label]
        if len(class_samples) < desired_class_size:
            resampled_class = resample(class_samples,
                                       replace=True,
                                       n_samples=desired_class_size,
                                       random_state=42)
        else:
            resampled_class = resample(class_samples,
                                       replace=False,
                                       n_samples=desired_class_size,
                                       random_state=42)

        X_resampled.append(resampled_class)
        y_resampled.extend([label] * len(resampled_class))

    X_resampled = np.vstack(X_resampled)
    y_resampled = np.array(y_resampled)

    resampled_label_counts = pd.Series(y_resampled).value_counts()
    print("Distribuția după balansare:\n", resampled_label_counts)

    X_train_balanced, X_test, y_train_balanced, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    print("Distribuția datelor de antrenament:")
    print(pd.Series(y_train_balanced).value_counts())
    print("Distribuția datelor de testare:")
    print(pd.Series(y_test).value_counts())

    return X_train_balanced, X_test, y_train_balanced, y_test

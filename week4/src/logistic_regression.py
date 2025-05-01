import numpy as np
from .utils import softmax, cross_entropy_loss


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def train(self, X, y):
        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        y_one_hot = np.eye(n_classes)[y]

        for i in range(self.epochs):
            logits = np.dot(X, self.weights) + self.bias
            probs = softmax(logits)
            error = probs - y_one_hot

            dw = np.dot(X.T, error) / n_samples
            db = np.sum(error, axis=0, keepdims=True) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = cross_entropy_loss(y_one_hot, probs)
            self.loss_history.append(loss)

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probs = softmax(logits)
        return np.argmax(probs, axis=1)

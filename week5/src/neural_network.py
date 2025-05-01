import numpy as np
from .utils import supervised_error, init_random, get_batches, shuffle, apply_activation


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1):
        self.weights = []
        self.biases = []
        self.learning_rate = learning_rate
        self.accuracy_history = []

        self.weights.append(init_random(layers[1], layers[0]))
        self.biases.append(init_random(layers[1], 1))

    def forward_propagation(self, instance):
        instance = np.reshape(instance, (len(instance), 1))
        current_input = instance
        for i in range(len(self.weights)):
            current_input = apply_activation(np.dot(self.weights[i], current_input) + self.biases[i])
        return current_input

    def train(self, train_set, epochs_len, batches_len):
        for i in range(epochs_len):
            train_set = shuffle(train_set)
            batches = get_batches(train_set, batches_len)
            error = 0
            for batch in batches:
                delta_w = np.zeros(self.weights[0].shape, dtype='float32')
                delta_b = np.zeros(self.biases[0].shape, dtype='float32')
                for instance in batch:
                    input = instance[0]
                    output = apply_activation(np.dot(self.weights[0], input) + self.biases[0])
                    error += supervised_error(instance[1], output)
                    delta_w += self.learning_rate * np.dot((instance[1] - output), instance[2])
                    delta_b += self.learning_rate * (instance[1] - output)
                self.weights[0] = self.weights[0] + delta_w
                self.biases[0] = self.biases[0] + delta_b
            error /= len(train_set)
            accuracy = int((1 - error) * 10000) / 100
            self.accuracy_history.append(accuracy)
            print(f"Epoch {i + 1}: {accuracy}% accuracy")

    def verify(self, dataset, name):
        errors = 0
        actual = []
        predicted = []

        for instance in dataset:
            output = self.forward_propagation(instance[0])
            errors += supervised_error(instance[1], output)

            actual.append(instance[1].argmax())
            predicted.append(output.argmax())
        error = errors / len(dataset)
        accuracy = int((1 - error) * 10000) / 100
        print(f"Accuracy for {name}: {accuracy}%")
        return actual, predicted

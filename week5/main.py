from src.neural_network import NeuralNetwork
from src.utils import load_data, convert, accuracy_plot, plot_confusion_matrix


def main():
    data_path = 'data/mnist.pkl.gz'

    train_set, valid_set, test_set = load_data(data_path)
    train_set = convert(train_set)
    valid_set = convert(valid_set)
    test_set = convert(test_set)

    neural_network = NeuralNetwork((28 * 28, 10), 0.01)
    neural_network.train(train_set, 30, 1000)
    neural_network.verify(valid_set, "Validation Set")
    actual, predicted = neural_network.verify(test_set, "Test Set")

    accuracy_plot(neural_network.accuracy_history)
    plot_confusion_matrix(actual, predicted, [i for i in range(10)])


if __name__ == '__main__':
    main()

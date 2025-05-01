import warnings
from src.utils import (load_data, preprocess_data, plot_loss, plot_confusion_matrix, plot_class_distribution,
                       class_distribution, balance_and_split_data)
from src.logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def main():
    X, y = load_data("data/symptom_diagnosis_dataset.csv")
    (X_train, X_test, y_train, y_test), encoder = preprocess_data(X, y)

    train_label_counts = class_distribution(y_train)
    plot_class_distribution(train_label_counts)
    (X_train, X_test, y_train, y_test) = balance_and_split_data(X_train, y_train, train_label_counts)

    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.2f}")

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    plot_loss(model.loss_history)
    plot_confusion_matrix(y_test, y_pred, encoder.classes_)


if __name__ == "__main__":
    main()

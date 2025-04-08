import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.knn import knn_predict, plot_misclassified_points, plot_classified_points


def main():
    df = pd.read_csv("data/dataset_hipertensiune.csv")
    X = df[["IMC", "Colesterol"]].values
    y = df["Hipertensiune"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_pred = knn_predict(X_train, y_train, X_test, k=5)
    accuracy = np.mean(y_pred == y_test)
    print(f"Acuratețea modelului kNN: {accuracy:.2f}")

    plot_classified_points(X_train, X_test, y_train, y_test)
    plot_misclassified_points(X_test, y_test, y_pred)


if __name__ == "__main__":
    main()

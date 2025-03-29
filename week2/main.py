import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.knn import knn_predict, plot_misclassified_points, plot_classified_points

df = pd.read_csv("data/dataset_hipertensiune.csv")
x = df[["IMC", "Colesterol"]].values
y = df["Hipertensiune"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_pred = knn_predict(x_train, y_train, x_test, k=5)

print("Accuracy:", accuracy_score(y_test, y_pred))

plot_misclassified_points(x_test, y_test, y_pred)
plot_classified_points(x_train, x_test, y_train, y_test)

from src.utils import evaluate, load_data, preprocess_data, split_data
from src.bayes import bayes_naive, bayes_optimal


def main():
	data_path = "data/Diabetes Classification.csv"
	data = load_data(data_path)
	print("Am incarcat datele")

	processed_data = preprocess_data(data)
	print("Am preprocesat datele")

	train_data, test_data = split_data(processed_data.values.tolist())
	print("Am împărțit datele în train și test")

	predictions = bayes_naive(train_data, test_data)
	print("Am aplicat Bayes Naive")

	accuracy_score = evaluate(predictions, test_data)
	print(f"Acuratețea Bayes Naive: {accuracy_score:.2f}%")

	predictions = bayes_optimal(train_data, test_data)
	print("Am aplicat Bayes Optimal")

	accuracy_score = evaluate(predictions, test_data)
	print(f"Acuratețea Bayes Optimal: {accuracy_score:.2f}%")


if __name__ == "__main__":
	main()

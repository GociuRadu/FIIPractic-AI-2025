from .utils import split_data, separate_by_class, summarize_dataset, calculate_class_probs


def bayes_naive(train_data, test_data):
    separated = separate_by_class(train_data)
    predictions = []
    summaries = {label: summarize_dataset(rows) for label, rows in separated.items()}
    for row in test_data:
        probs = calculate_class_probs(summaries, row[:-1])
        predictions.append(max(probs, key=probs.get))
    return predictions


def bayes_optimal(train_data, test_data):
    train_data_1, train_data_2 = split_data(train_data)
    separated_1 = separate_by_class(train_data_1)
    separated_2 = separate_by_class(train_data_2)
    summaries_1 = {label: summarize_dataset(rows) for label, rows in separated_1.items()}
    summaries_2 = {label: summarize_dataset(rows) for label, rows in separated_2.items()}
    predictions = []
    for row in test_data:
        probs_1 = calculate_class_probs(summaries_1, row[:-1])
        probs_2 = calculate_class_probs(summaries_2, row[:-1])
        final_probs = {}
        all_classes = set(probs_1) | set(probs_2)
        for class_value in all_classes:
            final_probs[class_value] = (probs_1.get(class_value, 0) + probs_2.get(class_value, 0)) / 2
        predictions.append(max(final_probs, key=final_probs.get))
    return predictions

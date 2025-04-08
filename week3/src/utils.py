import math
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def load_data(path):
    df = pd.read_csv(path)
    return shuffle(df, random_state=42)


def preprocess_data(df):
    df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'F' else 1)
    df.fillna(df.mean(), inplace=True)
    numeric_columns = ['Age', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN']
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    for col in numeric_columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df[col] = np.where(z_scores > 3, df[col].mean(), df[col])
    return df


def split_data(dataset, ratio=0.7):
    train_size = int(len(dataset) * ratio)
    train_data = random.sample(dataset, train_size)
    test_data = [item for item in dataset if item not in train_data]
    return train_data, test_data


def gaussian_prob(x, mean, stdev):
    if stdev == 0:
        return 1.0 if x == mean else 0.0
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        label = row[-1]
        if label not in separated:
            separated[label] = []
        separated[label].append(row[:-1])  # fără Diagnosis
    return separated


def summarize_dataset(data):
    summaries = []
    for col in zip(*data):
        mean = sum(col) / len(col)
        stddev = math.sqrt(sum((x - mean) ** 2 for x in col) / (len(col) - 1))
        summaries.append((mean, stddev))
    return summaries


def calculate_class_probs(summaries_by_class, row):
    probs = {}
    for class_value, summaries in summaries_by_class.items():
        probs[class_value] = 1
        for i in range(len(summaries)):
            mean, stdev = summaries[i]
            x = row[i]
            probs[class_value] *= gaussian_prob(x, mean, stdev)
    return probs


def evaluate(predictions, test_data):
    correct = sum(1 for p, y in zip(predictions, test_data) if p == y[-1])
    return correct / len(test_data) * 100

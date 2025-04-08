import numpy as np


def euclidean_distance(x1, x2):
	return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
	return np.sum(np.abs(x1 - x2))


def minkowski_distance(x1, x2, p=3):
	return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

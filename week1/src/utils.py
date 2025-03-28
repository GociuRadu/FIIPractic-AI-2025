import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def load_dataset(file_name):
    dataset_path = os.path.join("data", file_name)
    data = pd.read_csv(dataset_path)
    return shuffle(data, random_state=42)

def unique_values(data,column):
   return data[column].unique()

def split_dataset(data,column,value):
    subset1=data[data[column]==value]
    subset2=data[data[column]!=value]
    return subset1,subset2


def most_common_label(data,target):
    return data[target].mode()

def entropy(data,target):
    counts=data[target].value_counts(normalize=True)
    ent=-np.sum(counts*np.log2(counts))
    return ent
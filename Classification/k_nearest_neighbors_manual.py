from math import sqrt
import warnings
import numpy as np
from collections import Counter
from numpy.core.numeric import full
from numpy.lib.function_base import append
import pandas as pd
import random
import os

os.chdir("./Classification")


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!")
    # herein lies the problem with k nearest neighbors: the distance of the prediciton point to every other point must be calculated. This is expensive!!!
    distances = []
    for group in data:
        for features in data[group]:
            euclidian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidian_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / 5
    print(vote_result, confidence)
    return vote_result, confidence


df = (
    pd.read_csv("breast-cancer-wisconsin.data", na_values="?", header=0)
    .fillna(-99999)
    .drop("id", axis=1)
)
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size = 0.4
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[: -int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)) :]

[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1

print("Accuracy", correct / total)

import numpy as np
import pandas as pd

def gini(y):
    """
    menghitung gini impurity dari target y

    argument:
    y (array-like): array yang berisi nilai target klasifikasi

    output:
    float: gini impurity dari node

    rumus gini impurity
    Gini = 1 - (Pi^2 + pj^2)
    """

    K, counts = np.unique(y, return_counts = True)
    counts_unique = dict()

    for k, c in zip(K, counts):
        counts_unique[k] = c

    p_m = {}
    p_i = 0

    for k in K:
        p_m[k] = counts_unique[k] / len(y)
        p_i += (p_m[k]**2)

    impurity_node = 1 - p_i

    return impurity_node


def entropy(y):
    """
    menghitung gini impurity dari target y

    argument:
    y (array-like): array yang berisi nilai target klasifikasi

    output:
    float: entropy impurity dari node

    rumus entropy
    entropy(D) = - pi log2 (pi)
    """
    
    K, count = np.unique(y, return_counts=True)
    counts_unique = dict()

    for k, c in zip(K, count):
        counts_unique[k] = c

    p_m = {}
    p_i = 0

    for k in K:
        p_m[k] = counts_unique[k] / len(y)
        p_i += p_m[k] * np.log2(p_m[k])

    return -p_i

def log_loss(y):
    """
    menghitung log loss dari target y

    argument:
    y (array-like): array yang berisi nilai target klasifikasi

    output:
    float: log loss impurity dari node

    log loss di decision tree berbeda dengan yang di logistic regression.
    log loss disini digunakan sebagai metrik evaluasi.
    """

    K, counts = np.unique(y, return_counts=True)
    counts_unique = dict(zip(K, counts))

    p_m = {}

    for k in K:
        p_m[k] = counts_unique[k] / len(y)

    return 1 - p_m[np.argmax(counts)]

def mse(y):
    """
    menghitung mean squared error dari y untuk masalah regresi

    argument:
    y (array-like): array yang berisi nilai target regresi

    output:
    float: mse dari y
    """

    if len(y) == 0:
        return 0

    return np.mean((y - np.mean(y)) ** 2)

def mae(y):
    """
    menghitung mean absolute error dari y untuk masalah regresi

    argument:
    y (aray-like): array yang berisi nilai target regresi

    output:
    float: mae dari y
    """
    if len(y) == 0:
        return 0

    return np.mean(np.abs(y - np.mean(y)))


if __name__ == "__main__":
    data = pd.read_csv('apple_quality.csv')
    data['Quality'].fillna('bad')
    data['Quality'] = data['Quality'].apply(lambda x: 1 if x == "good" else 0)
    y = data.Quality

    y_example_1 = np.array([0] * 70 + [1] * 30)
    y_example_2 = np.array([0] * 65 + [1] * 35)
    y_example_3 = np.array([0] * 60 + [1] * 40)

    r = gini(y)
    print(r)
    rl = log_loss(y)
    print(rl)
    r_ent = entropy(y_example_3)
    print(r_ent)
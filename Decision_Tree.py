"""
referensi: https://github.com/axeltanjung/decision_tree_from_scratch/blob/main/ml_from_scratch/tree/_classes.py
"""

import numpy as np
from jupyter_core.version import parts
from xlrd.biffh import XL_RIGHTMARGIN

import CRITERIA as cr

CLF_CRITERIA = {
    'gini': cr.gini,
    'log_loss': cr.log_loss,
    'entropy': cr.entropy
}

REG_CRITERIA = {
    'mse': cr.mse,
    'mae': cr.mae
}

class Node:
    def __init__(self, feature = None,
                 threshold = None,
                 impurity = None,
                 samples = None,
                 values = None,
                 leaf= None,
                 left= None,
                 right= None):
        """
        :param feature (string): nama fitur atau kolom pada dataset
        :param threshold (float): titik potong pada fitur atau kolom
        :param impurity (float): perhitungan untuk mengukur apakah node tersebut tercampur dengan nilai yang berbeda atau tidak
        :param samples (int): jumlah data pada node
        :param values (array/list):nilai unik pada samples, penentu node tersebut murni atau tidak
        :param leaf: terminal node / node terakhir yang tidak memiliki cabang lagi
        :param left (class): cabang kiri pada node
        :param right (class): cabang kanan pada node
        """

        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.samples = samples
        self.values = values
        self.leaf = leaf
        self.left = left
        self.right = right
        
    def show_node_prop(self):
        print(f'{self.feature} = {self.threshold}\n'
              f'{self.impurity}\n'
              f'values = {self.values}')

def _split(data):
    """
    :param data: data yang akan dibagi, dimana data berupa array/list atau dataframe 
    :return: threshold yang digunakan untuk membagi data
    """
    val_unique = np.unique(data)
    n_shape = len(val_unique)
    val_unique.sort()
    
    thresh = np.zeros(n_shape - 1)
    
    for i in range(n_shape - 1):
        nilai_i = val_unique[i]
        nilai_ii = val_unique[i + 1]
        
        thresh[i] = (nilai_i + nilai_ii) / 2

    return thresh


def _split_data(data, fitur, thresh):
    data_thresh = data[:, fitur] <= thresh
    return data[data_thresh], data[~data_thresh]


class DecisionTreeBase:
    def __init__(self,
                 max_depth = None,
                 min_samples_split = None,
                 min_samples_leaf = None,
                 alpha = 0.0,
                 impurity_reduction = None,
                 impurity_type = 'gini'):
        """
        :param max_depth: seberapa dalam pohon akan membentuk cabang
        :param min_samples_split: menentukan jumlah minimal sampel yang ada pada dalam node internal agar dapat dipecah
        :param min_samples_leaf: menentukan jumlah minimal sampel yang ada pada cabang node
        :param alpha: fungsi cost pada pruning
        :param impurity_reduction: batas nilai impurity agar suatu threshold bisa dianggap sebagai node
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.impurity_reduction = impurity_reduction
        self.impurity = impurity_type
        self.n_feature = None
        self.n_samples = None
        self.tree_ = None

    def fit(self, X, y):
        cx = np.array(X).copy()
        cy = np.array(y).copy()
        self.n_samples, self.n_feature = X.shape
        self.columns = X.columns
        self._grow_tree(cx, cy)


    def _grow_tree(self, X, y):
        """
            langkah buat bentuk pohon:
            1. bikin fungsi buat misahin antar kolom dan nanti tiap kolom digabungin sama target (namanya xy)
            2. sorting setiap xy
            3. dari setiap xy, dicari nilai threshold dengan cara i ditambah i+1 bagi 2, dengan kriteria:
                - jika hasilnya sama seperti nilai i dan i+1, maka kosongi aja
                - jika hasilnya tidak sama seperti nilai i dan i+i, maka masukkan sebagai kandidat threshold
            4. hitung impurity dari kolom saat ini menggunakan threshold yang telah terbentuk sebelumnya dari kolom tersebut.
            5. setelah dapet semua impurity, ambil nilai impurity yang paling rendah sebagai node paling awal / root.
        """
        th_dict = self._part(X, y)

        # for i in sorted(th_dict, key=lambda x: th_dict[x][2]):
        #     print(f"feature: {th_dict[i][0]}, threshold: {th_dict[i][1]}, impurity: {th_dict[i][2]}")

        # for i in th_dict:
        #     print(f"feature: {th_dict[i][0]}, threshold: {th_dict[i][1]}, impurity: {th_dict[i][2]}")
        
        sort_th_dict = sorted(th_dict, key=lambda x: th_dict[x][2])
        first_val = th_dict[sort_th_dict[0]]
        print(sort_th_dict[0])
        print(first_val)

        # bikin root node dari first_val

    def _part(self, X, y):
        """
        fungsi ini berfokus pada pencarian nilai threshold, dan menghitung impurity dari nilai threshold tersebut.
        :param X: data fitur 
        :param y: data target
        :return: cabang root
        """
        thresh_dict = dict()
        feature = self.columns
        iter = 0

        # menghitung weighted average of gini impurity dari threshold yang dihasilkan oleh kolom
        for i in range(len(feature) - 1):
            X_iny = np.column_stack((X[:, i], y))
            sortX = np.argsort(X_iny[:, 0])
            X_iny = X_iny[sortX]
            X_i = X_iny[:, 0]
            thresh_i = _split(data=X_i)
            # print(thresh_i)
            for th_i in thresh_i:
                left, right = _split_data(X_iny, 0, th_i)
                # weighted average of gini impurity
                # print(f'feature {feature[i]}')
                # print(f"threshold: {th_i}")
                # print(f"left: \n{left}")
                # print("\n")
                # print(f"right: \n{right}")
                # print("\n")

                wagi = self._weighted_average_gini_impurity(left, right)
                thresh_dict[iter] = [feature[i], th_i, wagi]
                # print([feature[i], th_i, wagi])
                iter += 1
            # print("\n")
        
        return thresh_dict

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0

        unique_val, counts_val = np.unique(y, return_counts=True)
        probabilities = counts_val / len(y)
        sum_of_squares = np.sum(probabilities ** 2)
        impurity_node = 1 - sum_of_squares

        return impurity_node
    
    def _weighted_average_gini_impurity(self, left, right):
        y_left = left[:, -1]
        # print(f"y_left: {y_left}")
        y_right = right[:, -1]
        # print(f"y_right: {y_right}")
        left_gini = self._gini_impurity(y_left)
        # print(f"left_gini: {left_gini}")
        right_gini = self._gini_impurity(y_right)
        # print(f"right_gini: {right_gini}")
        # print("\n")
        # 
        wagi = ((len(y_left) / (len(y_left) + len(y_right))) * left_gini) + ((len(y_right) / (len(y_left) + len(y_right))) * right_gini)
        # print(wagi)
        return wagi


def count_majority_val(y):
    unique_val, counts_val = np.unique(y, return_counts=True)
    counts_max = np.argmax(counts_val)
    return unique_val[counts_max]

# debug
if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_csv('dolan.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    print(df)

    dt = DecisionTreeBase(max_depth=3, min_samples_split=2, min_samples_leaf=1)
    dt.fit(df.drop(['Play'], axis=1), df['Play'])

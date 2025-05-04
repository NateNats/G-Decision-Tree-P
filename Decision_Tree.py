"""
referensi: https://github.com/axeltanjung/decision_tree_from_scratch/blob/main/ml_from_scratch/tree/_classes.py
"""

import numpy as np
import collections

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
        print(f"Feature = {self.feature}")
        print(f"Threshold = {self.threshold}")
        print(f"Impurity = {self.impurity}")

        # Ringkas values (misal hitung distribusi kelas jika values berupa array)
        if self.values is not None:
            try:
                counter = collections.Counter(self.values)
                print(f"Class distribution = {dict(counter)}")
            except Exception:
                print(f"Values = {self.values}")
        else:
            print("Values = None")

        print(f"Samples = {self.samples}")
        print(f"Leaf = {self.leaf}")

        if self.left is not None:
            print(f"Left child: Feature = {self.left.feature}, Threshold = {self.left.threshold}")
        else:
            print("Left child: None")

        if self.right is not None:
            print(f"Right child: Feature = {self.right.feature}, Threshold = {self.right.threshold}")
        else:
            print("Right child: None")

        print()

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

    def print_tree(self, node=None, depth=0):
        if node is None:
            if depth == 0:
                print("The tree is empty, ensure the tree is properly trained using the fit method")
                return
            return

        indent = "  " * depth
        if node.leaf is not None:
            print(f"{indent}Leaf: Predict = {node.leaf}, Samples = {node.samples}")
        else:
            print(
                f"{indent}Node: Feature = {node.feature}, Threshold = {node.threshold:.4f}, Impurity = {node.impurity:.4f}, Samples = {node.samples}")
            print(f"{indent}Left:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}Right:")
            self.print_tree(node.right, depth + 1)

    def fit(self, X, y):
        cx = np.array(X).copy()
        cy = np.array(y).copy()
        self.n_samples, self.n_feature = X.shape
        self.columns = X.columns if hasattr(X, 'columns') else [str(i) for i in range(self.n_feature)]
        self.tree_ = self._grow_tree(cx, cy)

    def _grow_tree(self, X, y, depth = 0):
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

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (self.max_depth is not None) and (depth >= self.max_depth) or n_samples < self.min_samples_split or n_labels == 1:
            leaf_value = count_majority_val(y)
            return Node(leaf=leaf_value, samples=n_samples, values=np.bincount(y) if y.dtype.kind in 'iu' else y)

        th_dict = self._part(X, y)

        if not th_dict:
            leaf_value = count_majority_val(y)
            return Node(leaf=leaf_value, samples=n_samples, values=np.bincount(y) if y.dtype.kind in 'iu' else y)

        sorted_dict = sorted(th_dict, key=lambda x: th_dict[x][2])
        feature, threshold, impurity = th_dict[sorted_dict[0]]
        feature_index = list(self.columns).index(feature)

        left_idx = X[:, feature_index] <= threshold
        right_idx = X[:, feature_index] > threshold
        X_left, y_left = X[left_idx], y[left_idx]
        X_right, y_right = X[right_idx], y[right_idx]

        # menghandle kalau cabang kiri dan kanan kososng
        if len(y_left) == 0 or len(y_right) == 0:
            leaf_value = count_majority_val(y)
            node = Node(leaf = leaf_value, samples = n_samples, values = np.bincount(y) if y.dtype.kind in 'iu' else y)
            node.show_node_prop()
            return node

        # ngecek jumlah minimal sampel setiap cabang
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            leaf_value = count_majority_val(y)
            node = Node(leaf=leaf_value, samples=n_samples, values=np.bincount(y) if y.dtype.kind in 'iu' else y)
            node.show_node_prop()
            return node

        node = Node(feature=feature, threshold=threshold, impurity=impurity, samples=n_samples,
                    values=np.bincount(y) if y.dtype.kind in 'iu' else y)

        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        node.leaf = False
        node.show_node_prop()
        return node

    def _part(self, X, y):
        """
        fungsi ini berfokus pada pencarian nilai threshold, dan menghitung impurity dari nilai threshold tersebut.
        :param X: data fitur 
        :param y: data target
        :return: cabang root
        """
        thresh_dict = dict()
        feature = self.columns

        # menghitung weighted average of gini impurity dari threshold yang dihasilkan oleh kolom
        for i in range(len(feature)):
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
            # print("\n")
        
        return thresh_dict

    # def _best_split(self, thresh_dict):
    #     if not thresh_dict:
    #         return None
    #
    #     sorted_dict = sorted(thresh_dict.items(), key=lambda x: x[0][1][2])
    #     best_feature, best_key, best_val = sorted_dict[0]
    #     return best_feature, best_key, best_val
    #
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

    dt = DecisionTreeBase(max_depth = None, min_samples_split=2, min_samples_leaf=1)
    X_train, y_train = df.iloc[:13, :-1], df.iloc[:13, -1]
    dt.fit(X_train, y_train)

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    feature_vector_sorted = feature_vector[sorted_indices]
    target_vector_sorted = target_vector[sorted_indices]

    unique_values = np.unique(feature_vector_sorted)
    if len(unique_values) == 1:
        return None, None, None, None

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    ginis = []
    total_size = len(target_vector)

    for threshold in thresholds:
        left_mask = feature_vector_sorted < threshold
        right_mask = ~left_mask

        left_size = np.sum(left_mask)
        right_size = total_size - left_size

        if left_size == 0 or right_size == 0:
            continue

        p_left_1 = np.sum(target_vector_sorted[left_mask] == 1) / left_size
        p_left_0 = 1 - p_left_1
        gini_left = 1 - p_left_1 ** 2 - p_left_0 ** 2

        p_right_1 = np.sum(target_vector_sorted[right_mask] == 1) / right_size
        p_right_0 = 1 - p_right_1
        gini_right = 1 - p_right_1 ** 2 - p_right_0 ** 2
        gini_split = - (left_size / total_size) * gini_left - (right_size / total_size) * gini_right
        ginis.append(gini_split)

    ginis = np.array(ginis)

    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]) or len(sub_y) < self._min_samples_split or (self._max_depth is not None and depth == self._max_depth):
            node['type'] = 'terminal'
            node['class'] = Counter(sub_y).most_common(1)[0][0]
            return

        best_feature, best_value, best_gini = None, None, float('inf')
        num_features = sub_X.shape[1]

        for feature in range(num_features):
            feature_type = self._feature_types[feature]
            unique_values = np.unique(sub_X[:, feature])

            for value in unique_values:
                if self._is_valid_split(sub_X, feature, value):
                    gini = self._calculate_gini(sub_X, sub_y, feature, value)
                    if gini < best_gini:
                        best_feature, best_value, best_gini = feature, value, gini

        if best_feature is not None:
            node['type'] = 'nonterminal'
            node['feature_split'] = best_feature
            node['threshold'] = best_value
            node['left_child'], node['right_child'] = {}, {}

            left_indices = self._split_indices(sub_X, best_feature, best_value)
            self._fit_node(sub_X[left_indices], sub_y[left_indices], node['left_child'], depth + 1)
            self._fit_node(sub_X[~left_indices], sub_y[~left_indices], node['right_child'], depth + 1)

    def _is_valid_split(self, X, feature, value):
        split = (X[:, feature] == value) if self._feature_types[feature] == 'categorical' else (X[:, feature] < value)
        return np.sum(split) >= self._min_samples_leaf and np.sum(~split) >= self._min_samples_leaf

    def _calculate_gini(self, X, y, feature, value):
        split = (X[:, feature] == value) if self._feature_types[feature] == 'categorical' else (X[:, feature] < value)
        left_gini = self._gini(y[split])
        right_gini = self._gini(y[~split])
        left_weight = np.sum(split) / len(X)
        right_weight = np.sum(~split) / len(X)
        return left_gini * left_weight + right_gini * right_weight

    def _split_indices(self, X, feature, value):
        return (X[:, feature] == value) if self._feature_types[feature] == 'categorical' else (X[:, feature] < value)

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        feature_value = x[node['feature_split']]
        if self._feature_types[node['feature_split']] == 'categorical':
            if feature_value == node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            if feature_value < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(np.array(X), np.array(y), self._tree, 0)
        return self

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in np.array(X)])

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_params(self, deep=True):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


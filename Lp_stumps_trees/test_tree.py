import numpy as np
import data
from tree_ensemble import TreeEnsemble, LeafNode
from stump_ensemble import StumpEnsemble, Stump
import json
import time
import math
from box import Box

n_trees = 5  # total number of trees in the ensemble
dataset = "mnist_1_5"

X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[dataset]()
X_train, X_test = data.convert_to_float32(X_train), data.convert_to_float32(X_test)

X_train = X_train[:1000]
y_train = y_train[:1000]
# X_test = X_test[:5000]

eps = 0.6
order = 2
max_depth = 5
ensemble = TreeEnsemble('tree', X_test.shape[1], 1, 5, 5, 0, max_depth, 5, n_bins=40)
ensemble.pre_leaf_node_list.append(LeafNode(Box(), 0, dict(zip(range(X_train.shape[0]), np.zeros(X_train.shape[0])))))

for i in range(n_trees):
    begin = time.time()
    tree = ensemble.fit_tree_Lp(np.asarray(range(X_train.shape[0])), X_train, y_train, eps, 1, order = order, budget = np.zeros(y_train.shape[0]), box = Box())
    ensemble.add_weak_learner(tree)

    yf_test = y_test * ensemble.predict(X_test)
    print('Iteration: {}, test error: {:.2%}'.format(
        i, np.mean(yf_test < 0.0)))
    # tree = ensemble.fit_stumps_over_coords_Lp(np.asarray(range(X_train.shape[0])), X_train, y_train, np.zeros(y_train.shape[0]), [], eps, 1, 1, Box())
    ensemble.export_json('{}_{}_{}_tree_{}_{}_{}.json'.format(dataset, order, eps, i, X_train.shape[0], max_depth))

    print("Time: {}".format(time.time() - begin))

import numpy as np
import data
from tree_ensemble import TreeEnsemble
from stump_ensemble import StumpEnsemble, Stump
import json
import time
import math

# ensemble = StumpEnsemble('stump', list(range(2)), 1, 0)
# stump_1 = Stump(-0.5, 1, 0.5, 0, 0)
# stump_2 = Stump(-0.5, 1, 0.5, 1, 0)
# stump_3 = Stump(-0.5, 1, 0.7, 0, 0)

# ensemble.add_weak_learner(stump_1)
# ensemble.add_weak_learner(stump_2)
# # ensemble.add_weak_learner(stump_3)

# X_train = np.asarray([[0.7, 0.7]])
# y_train = np.asarray([1])
# res = ensemble.export_json()
# with open("model.json", "w+") as f:
#     json.dump(res, f)
# # print(ensemble.certify_exact_norm_zero(X_train, y_train, 1))
# print(ensemble.certify_Lp_bound(X_train, y_train, 0.3, precision = 0.01))

n_trees = 20  # total number of trees in the ensemble
model = 'robust_exact'  # robust tree ensemble

dataset = 'breast_cancer'
X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[dataset]()
X_train, X_test = data.convert_to_float32(X_train), data.convert_to_float32(X_test)

# eps = 0.05
print(eps)
eps = 1.0

# initialize a tree ensemble with some hyperparameters
ensemble = StumpEnsemble(weak_learner='stump', n_trials_coord=X_train.shape[1], 
                        lr=0.4, idx_clsf=0, n_bins=40)
# initialize gammas, per-example weights which are recalculated each iteration
gamma = np.ones(X_train.shape[0])
schedule_len = 5

# order = 1
# if order == 0:
#     eps = 1

# eps = 0.3
precision = 0.0001

for i in range(1, n_trees + 1):
    # fit a new tree in order to minimize the robust loss of the whole ensemble
    # precision = 0.0005
    if i < schedule_len:
        cur_eps = math.ceil((eps/schedule_len * (i + 1))/precision) * precision
    else:
        cur_eps = eps

    begin = time.time()
    weak_learner = ensemble.fit_stumps_over_coords(X_train, y_train, gamma, model, (cur_eps, 0.3), order = -2, precision=precision)
    print("time: {}".format(time.time() - begin))

    # print(ensemble.certify_exact_norm_zero(X_train, y_train, 1))
    # begin = time.time()
    margin_prev = ensemble.certify_exact(X_train, y_train, eps)  # needed for pruning
    # print(time.time() - begin)
    # print(margin_prev)
    ensemble.add_weak_learner(weak_learner)
    # ensemble.prune_last_tree(X_train, y_train, margin_prev, eps, model)
    # calculate per-example weights for the next iteration
    if model != 'plain':
        gamma = np.exp(-ensemble.certify_exact(X_train, y_train, eps))
    else:
        gamma = np.exp(-y_train*ensemble.predict(X_train))
    
    # track generalization and robustness
    yf_train = y_train * ensemble.predict(X_train)
    yf_test = y_test * ensemble.predict(X_test)

    min_yf_test = ensemble.certify_exact(X_test, y_test, eps)
    
    threshold_cumu_value = ensemble.build_cumu_threshold_value()
    # print(threshold_cumu_value)

    # if order == 0:
    #     min_yf_train = ensemble.certify_exact_norm_zero(X_train, y_train, eps)
    #     min_yf_test = ensemble.certify_exact_norm_zero(X_test, y_test, eps)
    # else:
    #     min_yf_train = ensemble.certify_Lp_bound(X_train, y_train, eps, threshold_cumu_value, order = 1, precision = precision)
    # # print(min_yf_test)
    #     # print(min_yf_train)
    #     min_yf_train = min_yf_train.min(axis = 1)

    y_min_L2 = ensemble.certify_Lp_bound(X_test, y_test, 0.5, threshold_cumu_value, order = 2, precision=precision, parallel = False)
    y_min_L1 = ensemble.certify_Lp_bound(X_test, y_test, 1.0, threshold_cumu_value, order = 1, precision=precision, parallel = False)
    y_min = ensemble.certify_exact(X_test, y_test, 0.3)
    
    y_min_L2 = y_min_L2.min(axis = 1)
    y_min_L1 = y_min_L1.min(axis = 1)

    # print(y_min, y_min_L1, y_min_L2)

    min_yf_test = np.min(np.concatenate((y_min.reshape([-1, 1]), y_min_L2.reshape([-1, 1]), y_min_L1.reshape([-1, 1])), axis = 1), axis = 1)
    # min_yf_test = ensemble.certify_Lp_bound(X_test, y_test, 1.0, None, order = -2, precision = 0.001)

    #     # print(min_yf_train)
    # min_yf_test = min_yf_test.min(axis = 1)

    loss = np.mean(np.exp(-min_yf_test))

    # min_L1_robust_yf_test = ensemble.certify_Lp_bound(X_test, y_test, eps, None, (), 1, 0.002)

    print('Iteration: {}, test error: {:.2%}, upper bound on robust test error: {:.2%}, loss: {:.5f}'.format(
        i, np.mean(yf_test < 0.0), np.mean(min_yf_test < 0.0), loss))

ensemble.export_json("{}_general_0.7.json".format(dataset))
ensemble.save("{}_general_0.7.ensemble".format(dataset))

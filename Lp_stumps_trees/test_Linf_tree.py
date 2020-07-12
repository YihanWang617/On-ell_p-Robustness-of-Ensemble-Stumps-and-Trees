import numpy as np
import data
from tree_ensemble import TreeEnsemble

n_trees = 2  # total number of trees in the ensemble
model = 'robust_bound'  # robust tree ensemble

dataset = 'breast_cancer'
X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[dataset]()
X_train, X_test = data.convert_to_float32(X_train), data.convert_to_float32(X_test)

print(X_train.shape)
X_train = X_train[:5000]
y_train = y_train[:5000]

# print(np.sum(y_train == 1))

# initialize a tree ensemble with some hyperparameters
ensemble = TreeEnsemble(weak_learner='tree', n_trials_coord=X_train.shape[1], 
                        lr=0.2, min_samples_split=5, min_samples_leaf=5, max_depth=5, 
                        max_weight=1.0, idx_clsf=0, n_bins = 40, schedule_len = 0)
# initialize gammas, per-example weights which are recalculated each iteration
gamma = np.ones(X_train.shape[0])
for i in range(1, n_trees + 1):
    # fit a new tree in order to minimize the robust loss of the whole ensemble
    weak_learner = ensemble.fit_tree(X_train, y_train, gamma, model, eps, depth=1)
    margin_prev = ensemble.certify_treewise(X_train, y_train, eps)  # needed for pruning
    ensemble.add_weak_learner(weak_learner)
    # ensemble.prune_last_tree(X_train, y_train, margin_prev, eps, model)
    # calculate per-example weights for the next iteration
    if model == 'plain':
        gamma = np.exp(-ensemble.predict(X_train) * y_train)
    else:
        gamma = np.exp(-ensemble.certify_treewise(X_train, y_train, eps))
    
    # track generalization and robustness
    yf_test = y_test * ensemble.predict(X_test)
    min_yf_test = ensemble.certify_treewise(X_test, y_test, eps)
    # if i == 1 or i % 5 == 0:
    print('Iteration: {}, test error: {:.2%}, upper bound on robust test error: {:.2%}'.format(
        i, np.mean(yf_test < 0.0), np.mean(min_yf_test < 0.0)))
    
    ensemble.export_json('{}_{}_{}_tree_{}.json'.format(dataset, eps, model, i))

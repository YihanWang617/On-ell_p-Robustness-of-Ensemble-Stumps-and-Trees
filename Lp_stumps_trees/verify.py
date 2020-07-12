'''
this file is used to verify a stump ensemble
'''

from stump_ensemble import Stump, StumpEnsemble
import numpy as np
import data
import time
import argparse

def verify(X, y, model_path, eps, num_trials = 10, order = 0, precision = 0.02, coords_to_ignore = ()):
    ensemble = StumpEnsemble(weak_learner='stump', n_trials_coord = X.shape[1], lr = 0, idx_clsf = 0)
    with open(model_path, 'rb') as f:
        ensemble_arr = np.load(f, allow_pickle = False)
        # ensemble_arr = ensemble_arr[:20]
        ensemble.load(ensemble_arr)
    print(len(ensemble.trees))
    # for tree in ensemble.trees:
    #     print(tree.w_l, tree.w_r)
    # start = time.time()
    num, dim = X.shape
    sum_bound = 0
    # ori_pred = y * ensemble.predict(X)
    ori_pred = np.zeros(y.shape)

    if order == 0:
        eps = int(eps)

    robust_bound = np.zeros_like(y)

    threshold_cumu_value = ensemble.build_cumu_threshold_value()

    num_success = 0
    # start = time.time()
    begin = time.time()
    for i in range(num):
        # print(X[i])
        # print(time.time() - begin)
        if i == 1:
            begin = time.time()
        # print(time.time() - begin)
        eps_low = 0
        eps_high = None
        start_eps = eps

        # begin = time.time()
        for j in range(num_trials):
            # print(j)
            if eps_high is None:
                cur_eps = start_eps
            else:
                cur_eps = (eps_low + eps_high)/2
                if order == 0:
                    cur_eps = int(cur_eps)
            if order == 0:
                if cur_eps < 1:
                    cur_eps = 0
                    y_min = np.inf
                else:
                    y_min = ensemble.certify_exact_norm_zero(X[i].reshape([1, -1]), y[i], cur_eps)
            elif order == -2:
                diff_eps = (0.3, 0.2, 0.1)
                if cur_eps >= precision:
                    y_min_L2 = ensemble.certify_Lp_bound(X[i].reshape([1, -1]), y[i].reshape([1]), cur_eps * diff_eps[1], threshold_cumu_value, order = 2, precision=precision, parallel = False)
                    y_min_L1 = ensemble.certify_Lp_bound(X[i].reshape([1, -1]), y[i].reshape([1]), cur_eps * diff_eps[0], threshold_cumu_value, order = 1, precision=precision, parallel = False)
                    y_min = ensemble.certify_exact(X[i].reshape([1, -1]), y[i].reshape([1]), cur_eps * diff_eps[2])
                    
                    y_min_L2 = y_min_L2.min(axis = 1)
                    y_min_L1 = y_min_L1.min(axis = 1)
                else:
                    y_min_L2 = ensemble.certify_Lp_bound(X[i].reshape([1, -1]), y[i].reshape([1]), precision * diff_eps[1], threshold_cumu_value, order = 2, precision=precision, parallel = False)
                    y_min_L1 = ensemble.certify_Lp_bound(X[i].reshape([1, -1]), y[i].reshape([1]), precision * diff_eps[0], threshold_cumu_value, order = 1, precision=precision, parallel = False)
                    y_min = ensemble.certify_exact(X[i].reshape([1, -1]), y[i].reshape([1]), precision * diff_eps[2])

                    y_min_L2 = y_min_L2.min(axis = 1)
                    y_min_L1 = y_min_L1.min(axis = 1)

                    cur_eps = 0
                y_min = min(y_min, y_min_L2, y_min_L1)
            elif order > 0:
                if cur_eps >= precision:
                    # begin = time.time()
                    y_min = ensemble.certify_Lp_bound(X[i].reshape([1, -1]), y[i].reshape([1]), cur_eps, threshold_cumu_value, order = order, precision=precision, parallel = False)
                    # print(time.time() - begin)
                else:
                    y_min = ensemble.certify_Lp_bound(X[i].reshape([1, -1]), y[i].reshape([1]), precision, threshold_cumu_value, order = order, precision=precision, parallel = False)
                    cur_eps = 0
            # begin = time.time()
                y_min = y_min.min(axis = 1)
            else:
                if cur_eps >= precision:
                    # begin = time.time()
                    y_min = ensemble.certify_exact(X[i].reshape([1, -1]), y[i].reshape([1]), cur_eps) #(X[i].reshape([1, -1]), y[i].reshape([1]), cur_eps, threshold_cumu_value, order = order, precision=precision, parallel = False)
                    # print(time.time() - begin)
                else:
                    y_min = ensemble.certify_exact(X[i].reshape([1, -1]), y[i].reshape([1]), precision)#ensemble.certify_Lp_bound(X[i].reshape([1, -1]), y[i].reshape([1]), cur_eps, threshold_cumu_value, order = order, precision=precision, parallel = False)
                    cur_eps = 0
            # print(time.time() - begin)
            # y_min = ori_pred[i] + y_min

            # print(y_min, cur_eps, j)
            if y_min < 0:
                eps_high = cur_eps
            else:
                if j == 0:
                    num_success += 1
                if eps_high is None:
                    start_eps *= 2
                else:
                    eps_low = cur_eps
        # print("end of verification: {}:{}".format(eps_low, eps_high))
        if eps_high is None:
            # print(start_eps)
            # robust_bound[i] = start_eps
            sum_bound += start_eps
        else:
            # print(eps_low)
            # robust_bound[i] = eps_low
            sum_bound += eps_low
        # print(time.time() - begin)
    print("avg. time: {}".format((time.time() - begin)/(num - 1)))

    min_yf_test = y * ensemble.predict(X)
    print("Average bound of the dataset is {}, robust error: {}, standard error: {}, num_samples: {}".format(sum_bound/num, 1 - num_success/num, np.mean(min_yf_test < 0), num))
    # print("time span: {}".format(time.time() - start))
    return robust_bound
        
                    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default = 'breast_cancer', help="dataset to be verified")
    parser.add_argument('--model', type=str, default = 'breast_cancer.ensemble.npy', help='model name to be verified')
    parser.add_argument('--precision', type=float, default=0.0001, help='precision of the verification')
    parser.add_argument('--eps', type=float, default=0.3, help='initial epsilon')
    parser.add_argument('--num_trials', type=int, default=10, help='number of trials')
    parser.add_argument('--order', type=int, default = 1, help='order')

    args = parser.parse_args()
    print(args)

    X_train, y_train, X_test, y_test, _= data.all_datasets_dict[args.dataset]()

    # start = time.time()
    verify(X_test, y_test, args.model, args.eps, args.num_trials, args.order, precision=args.precision)
    # print("time span: {}".format(time.time() - start))
    # print(X_test)
            
    
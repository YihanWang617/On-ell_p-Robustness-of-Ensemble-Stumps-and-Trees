import numpy as np
import argparse
import data
from stump_ensemble import StumpEnsemble, Stump
from tree_ensemble import TreeEnsemble, Tree, LeafNode
import math
import time
from box import Box


def main():
    np.random.seed(1)
    np.set_printoptions(precision=10)

    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='breast_cancer, diabetes, cod_rna, mnist_1_5, mnist_2_6, fmnist_sandal_sneaker, gts_30_70,'
                             ' gts_100_roadworks')
    # parser.add_argument('--model', type=str, default='plain',
                        # help='plain, da_uniform, at_cube, robust_exact, robust_bound.')
    parser.add_argument('--weak_learner', type=str, default='stump', help='stump or tree')
    parser.add_argument('--max_depth', type=int, default=4, help='Depth of trees (only used when weak_learner==tree).')
    parser.add_argument('--max_weight', type=float, default=1.0, help='The maximum leaf weight.')
    parser.add_argument('--n_bins', type=int, default=-1, help='By default we check all thresholds.')
    parser.add_argument('--lr', type=float, default=0.2, help='Shrinkage parameter (aka learning rate).')
    # parser.add_argument('--eps', type=float, default=-1, help='Linf epsilon. -1 means to use the default epsilons.')
    parser.add_argument('--n_train', type=int, default=-1, help='Number of training points to take.')
    parser.add_argument('--n_tree', type=int, default=20, help='Number of trees or stumps.')
    parser.add_argument('--precision', type=float, default=0.02, help='precision of verification.')
    parser.add_argument('--order', type=int, default=1, help='order of the model.')
    parser.add_argument('--schedule_len', type=int, default=4, help='')
    parser.add_argument('--sample_size', type=int, default=-1)
    # parser.add_argument('--debug', action='store_true', help='Debugging mode: not many samples for the attack.')
    args = parser.parse_args()
    print(args)

    n_trees = args.n_tree
    X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[args.dataset]()
    X_train, X_test = data.convert_to_float32(X_train), data.convert_to_float32(X_test)

    if args.sample_size > 0:
        X_train = X_train[:args.sample_size]
        y_train = y_train[:args.sample_size]

    print("training the {} ensemble on {}, eps = {}".format(args.weak_learner, args.dataset, eps))

    if args.weak_learner == 'stump':
        ensemble = StumpEnsemble(weak_learner=args.weak_learner, n_trials_coord=X_train.shape[1], 
                        lr=args.lr, idx_clsf=0, n_bins=args.n_bins, max_weight=args.max_weight)
    else:
        ensemble = TreeEnsemble(args.weak_learner, X_train.shape[1], args.lr, 5, 5, 0, args.max_depth, args.schedule_len, max_weight=args.max_weight, n_bins=args.n_bins)
        ensemble.pre_leaf_node_list.append(LeafNode(Box(), 0, dict(zip(range(X_train.shape[0]), np.zeros(X_train.shape[0])))))

    schedule_len = args.schedule_len if args.weak_learner == 'stump' else 0
    # order = 1
    if args.order == 0:
        eps = 1

    gamma = np.ones(X_train.shape[0])
    for i in range(1, n_trees + 1):
        # fit a new tree in order to minimize the robust loss of the whole ensemble
        precision = 0.005
        if i < schedule_len:
            cur_eps = math.ceil((eps/schedule_len * (i + 1))/precision) * precision
        else:
            cur_eps = eps

        model = 'robust_exact' if args.weak_learner == 'stump' else 'robust_bound'
        begin = time.time()
        if args.weak_learner == 'stump':
            weak_learner = ensemble.fit_stumps_over_coords(X_train, y_train, gamma, model, cur_eps, order = args.order,  precision = args.precision)
            margin_prev = ensemble.certify_exact(X_train, y_train, eps)
            gamma = np.exp(-ensemble.certify_exact(X_train, y_train, eps))
        else:
            weak_learner = ensemble.fit_tree_Lp(np.asarray(range(X_train.shape[0])), X_train, y_train, cur_eps, 1, args.order, np.zeros(X_train.shape[0]))
        print("time: {}".format(time.time() - begin))

        # print(ensemble.certify_exact_norm_zero(X_train, y_train, 1))
        # begin = time.time()
        # margin_prev = ensemble.certify_exact(X_train, y_train, eps)  # needed for pruning
        # print(time.time() - begin)
        # print(margin_prev)
        ensemble.add_weak_learner(weak_learner)
        # ensemble.prune_last_tree(X_train, y_train, margin_prev, eps, model)
        # calculate per-example weights for the next iteration
        # gamma = np.exp(-ensemble.certify_exact(X_train, y_train, eps))
        
        # track generalization and robustness
        yf_train = y_train * ensemble.predict(X_train)
        yf_test = y_test * ensemble.predict(X_test)
        
        # threshold_cumu_value = ensemble.build_cumu_threshold_value()
        # print(threshold_cumu_value)


        if args.weak_learner == 'stump':
            if args.order == 0:
                min_yf_train = ensemble.certify_exact_norm_zero(X_train, y_train, eps)
                min_yf_test = ensemble.certify_exact_norm_zero(X_test, y_test, eps)
            else:
                min_yf_train = ensemble.certify_Lp_bound(X_train, y_train, eps, threshold_cumu_value, order = 1, precision = precision)
            # print(min_yf_test)
                # print(min_yf_train)
                min_yf_train = min_yf_train.min(axis = 1)

                min_yf_test = ensemble.certify_Lp_bound(X_test, y_test, eps, threshold_cumu_value, order = 1, precision = precision)

                # print(min_yf_train)
                min_yf_test = min_yf_test.min(axis = 1)

            min_yf_test_Linf = ensemble.certify_treewise(X_test, y_test, eps)

            loss = np.mean(np.exp(-min_yf_train))

            print('Iteration: {}, test error: {:.2%}, upper bound on robust test error: {:.2%}, upper bound on Linf robust error: {:.2%}, loss: {:.5f}'.format(
                i, np.mean(yf_test < 0.0), np.mean(min_yf_test < 0.0), np.mean(min_yf_test_Linf < 0.0), loss))
        else:
            min_yf_test_Linf = ensemble.certify_treewise(X_test, y_test, eps)
            # min_yf_train = ensemble.certify_treewise(X_train, y_test, eps)
            yf_test = ensemble.predict(X_test) * y_test
            print('Iteration: {}, test error: {:.2%}, upper bound on Linf robust error: {:.2%}'.format(
                i, np.mean(yf_test < 0.0), np.mean(min_yf_test_Linf < 0.0)))

    ensemble.export_json("{}_{}_{}.json".format(args.dataset, eps, args.weak_learner))
    ensemble.save("{}_{}_{}.ensemble".format(args.dataset, eps, args.weak_learner))

if __name__== '__main__':
    main()
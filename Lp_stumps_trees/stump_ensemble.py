import numpy as np
from numba import jit
from collections import OrderedDict
from robust_boosting import exp_loss_robust, dtype, fit_plain_stumps, fit_robust_bound_stumps, fit_robust_exact_stumps, fit_robust_exact_stumps_Lp, fit_robust_exact_stumps_L0, fit_robust_exact_stumps_universal, certify_Lp_bound_iter_feature, certify_Lp_bound_numba
from utils import minimum, get_contiguous_indices, get_n_proc
from concurrent.futures import ThreadPoolExecutor
import math
import json
import time


class Stump:
    def __init__(self, w_l, w_r, b, coord, loss):
        # `loss` is the loss of the whole ensemble after applying this stump
        self.w_l, self.w_r, self.b, self.coord, self.loss = w_l, w_r, b, coord, loss
        self.left, self.right = None, None

    def predict(self, X):
        value = self.w_l + self.w_r * (X[:, self.coord] >= self.b)
        return value

    def find_min_yf(self, X, y, eps):
        split_lbs, split_ubs = X[:, self.coord] - eps, X[:, self.coord] + eps
        lval, rval = self.w_l, self.w_r + self.w_l

        # Fast vectorized version
        guaranteed_left = split_ubs < self.b
        guaranteed_right = split_lbs > self.b
        uncertain = (split_lbs <= self.b) * (split_ubs >= self.b)

        lbs = y*lval * guaranteed_left + y*rval * guaranteed_right + np.minimum(y*lval, y*rval) * uncertain

        return lbs

    def __repr__(self):
        lval, rval, threshold = self.w_l, self.w_r + self.w_l, self.b
        return 'Tree: if x[{}] < {:.4f}: {:.4f} else {:.4f}'.format(self.coord, threshold, lval, rval)

    def get_json_dict(self, counter_terminal_nodes):
        """
        counter_terminal_nodes: not used here
        """
        precision = 5
        children_list = [{'nodeid': 1, 'leaf': round(self.w_l, precision)},
                         {'nodeid': 2, 'leaf': round(self.w_l + self.w_r, precision)}]
        stump_dict = {'nodeid': 0, 'split': 'f' + str(int(self.coord)), 'split_condition': round(self.b, precision),
                      'yes': 1, 'no': 2, 'children': children_list}

        return stump_dict, counter_terminal_nodes


class StumpEnsemble:
    def __init__(self, weak_learner, n_trials_coord, lr, idx_clsf, n_bins=-1, max_weight=1.0):
        self.weak_learner = weak_learner
        self.n_trials_coord = n_trials_coord
        self.lr = lr
        self.idx_clsf = idx_clsf
        self.n_bins = n_bins
        self.max_weight = max_weight
        self.trees = []
        self.coords_trees = OrderedDict()

    def __repr__(self):
        sorted_trees = sorted(self.trees, key=lambda tree: tree.coord)
        return '\n'.join([str(t) for t in sorted_trees])

    def copy(self):
        ensemble_new = StumpEnsemble(self.weak_learner, self.n_trials_coord, self.lr, self.idx_clsf, self.n_bins,
                                     self.max_weight)
        for tree in self.trees:
            ensemble_new.add_weak_learner(tree, apply_lr=False)
        return ensemble_new

    def load(self, ensemble_arr, iteration=-1):
        if iteration != -1:  # take up to some iteration
            ensemble_arr = ensemble_arr[:iteration+1]
        for i in range(ensemble_arr.shape[0]):
            w_l, w_r, b, coord, loss = ensemble_arr[i, :]
            coord = int(coord)
            tree = Stump(w_l, w_r, b, coord, loss)
            # the values of w_l and w_r should be already scaled by lr, would be wrong to do this again
            self.add_weak_learner(tree, apply_lr=False)

    def export_model(self):
        ensemble_arr = np.zeros([len(self.trees), 5])
        for i, tree in enumerate(self.trees):
            ensemble_arr[i, :] = [tree.w_l, tree.w_r, tree.b, tree.coord, tree.loss]
        # print(ensemble_arr)
        return ensemble_arr

    # conver the stump ensemble to xgboost json
    def export_json(self, filename = ""):
        res = []
        for i, tree in enumerate(self.trees):
            obj = {"nodeid": 0, "depth": 0, "split": tree.coord, "split_condition": float(tree.b), "yes": 1, "no": 2, "missing": 2, "children":[
                {"nodeid": 1, "leaf": float(tree.w_l)},
                {"nodeid": 2, "leaf": float(tree.w_l + tree.w_r)},
            ]}
            res.append(obj)
        if filename != "":
            with open(filename, "w+") as f:
                json.dump(res, f)
        return res

    def save(self, path):
        if path != '':
            np.save(path, self.export_model(), allow_pickle=False)

    def add_weak_learner(self, tree, apply_lr=True):
        if apply_lr:
            tree.w_l, tree.w_r = tree.w_l*self.lr, tree.w_r*self.lr
        self.trees.append(tree)
        if tree.coord not in self.coords_trees:
            self.coords_trees[tree.coord] = []
        self.coords_trees[tree.coord].append(tree)

    def add_empty_weak_learner(self):
        empty_stump = Stump(0.0, 0.0, 0.0, 0, 0.0)
        self.add_weak_learner(empty_stump)

    def predict(self, X, coord_ignore = ()):
        Fx = np.zeros(X.shape[0])
        for tree in self.trees:
            if tree.coord in coord_ignore:
                continue
            Fx += tree.predict(X)
        return Fx

    def attack_by_sampling(self, X, y, eps, n_trials):
        """ A simple attack just by sampling in the Linf-box around the points. More of a sanity check. """
        num, dim = X.shape
        f_x_vals = np.zeros((num, n_trials))
        # Note: for efficiency, we sample the same random direction for all points
        deltas = np.random.uniform(-eps, eps, size=(dim, n_trials))
        for i in range(n_trials-1):
            # let's keep them as real images, although not strictly needed
            perturbed_pts = np.clip(X + deltas[:, i], 0.0, 1.0)
            f_x_vals[:, i] = self.predict(perturbed_pts)
        # maybe in some corner cases, the predictions at the original point is more worst-case than the sampled points
        f_x_vals[:, n_trials-1] = self.predict(X)

        f_x_min = np.min(y[:, None] * f_x_vals, axis=1)
        return f_x_min

    def certify_treewise(self, X, y, eps):
        lb_ensemble = np.zeros(X.shape[0])

        # The naive tree-wise bounded on the merged trees
        for tree in self.trees:
            lb_ensemble += tree.find_min_yf(X, y, eps)
        return lb_ensemble

    @staticmethod
    @jit(nopython=True)
    def find_min_coord_diff(X_proj, y, thresholds, w_r_values, eps):
        # parallel=True doesn't help here; not sure if jit here is helpful at all. maybe if there are many thresholds
        num = X_proj.shape[0]
        idx = np.argsort(thresholds)
        sorted_thresholds = thresholds[idx]
        sorted_w_r = w_r_values[idx]
        f_x_min_coord_diff, f_x_cumsum = np.zeros(num), np.zeros(num)
        for i_t in range(len(sorted_thresholds)):
            # consider the threshold if it belongs to (x-eps, x+eps] (x-eps is excluded since already evaluated)
            idx_x_eps_close_to_threshold = (X_proj - eps < sorted_thresholds[i_t]) * (sorted_thresholds[i_t] <= X_proj + eps)
            f_diff = y * sorted_w_r[i_t] * idx_x_eps_close_to_threshold
            f_x_cumsum += f_diff
            f_x_min_coord_diff = minimum(f_x_cumsum, f_x_min_coord_diff)
        return f_x_min_coord_diff

    def build_cumu_threshold_value(self):
        threshold_cumu_value = {}
        for coord in self.coords_trees.keys():
            trees_current_coord = self.coords_trees[coord]

            pred_base = 0
            thresholds, w_r_values = np.zeros(len(trees_current_coord)), np.zeros(len(trees_current_coord))
            for i in range(len(trees_current_coord)):
                tree = trees_current_coord[i]
                pred_base += tree.w_l
                thresholds[i], w_r_values[i] = tree.b, tree.w_r

            # print(thresholds, coord)
            # merge trees with the same thresholds to prevent an overestimation (lower bounding) of the true minimum
            thresholds_list, w_r_values_list = [], []
            for threshold in np.unique(thresholds):
                thresholds_list.append(threshold)
                w_r_values_list.append(np.sum(w_r_values[thresholds == threshold]))

            sorted_values = sorted(list(zip(thresholds_list, w_r_values_list)), key=lambda x: x[0])
            sorted_values.insert(0, (-np.inf, 0))
            sorted_values.append((np.inf, np.inf))

            cumu_value = []
            pre = 0
            for index, (b, value) in enumerate(sorted_values[:-1]):
                pre += value
                cumu_value.append((b, sorted_values[index + 1][0], pre + pred_base))
            threshold_cumu_value[coord] = cumu_value
        return threshold_cumu_value

    def certify_Lp_bound_batch(self, X, y, threshold_cumu_value, coords_to_ignore, order, precision, C):

        num, dim = X.shape

        min_diff = np.zeros((y.shape[0], C + 1))

        pre_dp = np.zeros(C + 1)
        dp = np.zeros(C + 1)

        for i in range(num):
            pre_dp.fill(0)
            dp.fill(0)

            y_sig = int(y[i])
            X_value = X[i]

            for coord, cumu_value in threshold_cumu_value.items():
                # dp, pre_dp = pre_dp, dp
                
                X_proj = X_value[coord]
                if coord in coords_to_ignore:
                    continue
                # cumu_value = threshold_cumu_value[coord]

                intervals = tuple([(i[0], i[1]) for i in cumu_value])
                values = tuple([i[2] for i in cumu_value])

                # print(intervals, values)
                # exit()

                dp = certify_Lp_bound_iter_feature(pre_dp, dp, X_proj, y_sig, intervals, values, order, precision, C)
                # --------------
                # ---------------
                # pre_dp = np.copy(dp)
                dp, pre_dp = pre_dp, dp
            min_diff[i] = np.copy(pre_dp)
        return min_diff

    def certify_Lp_bound(self, X, y, eps, threshold_cumu_value, coords_to_ignore = (), order = 1, precision = 0.0001, parallel = False):
        parallel = False
        assert type(order) == int, "the order should be an integer"

        C = math.ceil((eps**order)/precision)

        if precision > eps:
            return np.zeros((y.shape[0], C + 1)) + np.inf
        assert precision <= eps, 'percision is larger than eps'

        if threshold_cumu_value is None:
            threshold_cumu_value = self.build_cumu_threshold_value()

        num, dim = X.shape

        min_diff = np.zeros((y.shape[0], C + 1))

        pre_dp = np.zeros(C + 1)
        dp = np.zeros(C + 1)

        # dp_matrix = np.zeros((len(self.coords_trees), C + 1))
        if parallel:
            n_proc = 16
            # n_proc = min(n_coords, min(100, n_proc))
            batch_size = num // n_proc
            n_batches = num // batch_size + 1

            with ThreadPoolExecutor(max_workers=n_proc) as executor:
                procs = []
                for i_batch in range(n_batches):
                    # coords = features_to_check[i_batch*batch_size:(i_batch+1)*batch_size]
                    args = (X[i_batch*batch_size:(i_batch+1)*batch_size], y[i_batch*batch_size:(i_batch+1)*batch_size], threshold_cumu_value, coords_to_ignore, order, precision, C)
                    procs.append(executor.submit(self.certify_Lp_bound_batch, *args))

                # Process the results
                # i_coord = 0
                for i_batch in range(n_batches):
                    res_many = procs[i_batch].result()
                    # print(res_many)
                    # print(res_many.shape)
                    # for res in res_many:
                    min_diff[i_batch*batch_size:(i_batch+1)*batch_size] = res_many
                        # i_coord += 1
        else:
            # begin = time.time()
            for i in range(num):
                pre_dp.fill(0)
                dp.fill(0)

                y_sig = int(y[i])
                X_value = X[i]

                for coord, cumu_value in threshold_cumu_value.items():
                    # dp, pre_dp = pre_dp, dp
                    
                    X_proj = X_value[coord]
                    if coord in coords_to_ignore:
                        continue
                    # cumu_value = threshold_cumu_value[coord]

                    intervals = tuple([(i[0], i[1]) for i in cumu_value])
                    values = tuple([i[2] for i in cumu_value])

                    dp = certify_Lp_bound_iter_feature(pre_dp, dp, X_proj, y_sig, intervals, values, order, precision, C)
                    # --------------
                    # ---------------
                    # pre_dp = np.copy(dp)
                    dp, pre_dp = pre_dp, dp
            # min_diff = certify_Lp_bound_numba(pre_dp, dp, X, y, threshold_cumu_value, order, precision, C, coords_to_ignore)
                min_diff[i] = np.copy(pre_dp)
                # min_diff[i] = np.copy(pre_dp)
            # print("time: {}".format(time.time() - begin))

        return min_diff
      

    def certify_exact_norm_zero(self, X, y, eps, coords_to_ignore = ()):
        assert type(eps) == int, 'eps in L-0 verification should be a integer'

        num, dim = X.shape
        max_diff = np.zeros((dim, num))
        f_x_min = np.zeros(num)

        for coord in self.coords_trees.keys():
            if coord in coords_to_ignore:
                continue
            trees_current_coord = self.coords_trees[coord]

            pred_base = 0
            thresholds, w_r_values = np.zeros(len(trees_current_coord)), np.zeros(len(trees_current_coord))
            for i in range(len(trees_current_coord)):
                tree = trees_current_coord[i]
                pred_base += tree.w_l
                thresholds[i], w_r_values[i] = tree.b, tree.w_r
            
            # print(thresholds, w_r_values)

            # merge trees with the same thresholds to prevent an overestimation (lower bounding) of the true minimum
            thresholds_list, w_r_values_list = [], []
            for threshold in np.unique(thresholds):
                thresholds_list.append(threshold)
                w_r_values_list.append(np.sum(w_r_values[thresholds == threshold]))

            sorted_values = sorted(list(zip(thresholds_list, w_r_values_list)), key=lambda x: x[0])
            # print(coord, sorted_values)

            ori_pred = np.zeros(num)
            for tree in trees_current_coord:
                ori_pred += tree.predict(X)
            
            f_x_min += ori_pred
            # print(ori_pred)
            ori_pred *= y
            # print(pred_base)
            min_value = y * pred_base
            for threshold, w_r in sorted_values:
                pred_base += w_r
                min_value = np.minimum(min_value, y * pred_base)
            # print(ori_pred)
            # print(min_value)
            max_diff_value = ori_pred - min_value # max diff value of this coordinate for each sample
            max_diff[coord] = max_diff_value

        # print(max_diff)
        index_sort_coord = np.argsort(max_diff, axis = 0)
        # print(index_sort_coord[-1:, :])
        index_sort_coord = index_sort_coord[-int(eps):, :]
        # print(index_sort_coord)

        # print(f_x_min)
        f_x_min *= y
        for i in range(num):
            coord_diff = np.sum(max_diff[index_sort_coord[:, i], i])
            # print(coord_diff)
            f_x_min[i] -= coord_diff

        return f_x_min            
            # thresholds, w_r_values = np.array(thresholds_list), np.array(w_r_values_list)


    def certify_exact(self, X, y, eps, coords_to_ignore=()):
        # Idea: iterate over all thresholds, and then check if they are in (x-eps, x+eps]
        num, dim = X.shape
        f_x_min = np.zeros(num)

        # Fast, vectorized version
        for coord in self.coords_trees.keys():
            if coord in coords_to_ignore:
                continue
            trees_current_coord = self.coords_trees[coord]

            f_x_min_coord_base = np.zeros(num)
            thresholds, w_r_values = np.zeros(len(trees_current_coord)), np.zeros(len(trees_current_coord))
            for i in range(len(trees_current_coord)):
                tree = trees_current_coord[i]
                f_x_min_coord_base += y * tree.predict(X - eps)
                thresholds[i], w_r_values[i] = tree.b, tree.w_r

            # merge trees with the same thresholds to prevent an overestimation (lower bounding) of the true minimum
            thresholds_list, w_r_values_list = [], []
            for threshold in np.unique(thresholds):
                thresholds_list.append(threshold)
                w_r_values_list.append(np.sum(w_r_values[thresholds == threshold]))
            thresholds, w_r_values = np.array(thresholds_list), np.array(w_r_values_list)

            f_x_min += f_x_min_coord_base + self.find_min_coord_diff(X[:, coord], y, thresholds, w_r_values, eps)
        return f_x_min

    def fit_stumps_over_coords(self, X, y, gamma, model, eps, order = -1, precision = 0.02, coords_to_ignore = ()):
        verbose = True
        parallel = False  # can speed up the training on large datasets
        n_ex = X.shape[0]
        # X, y, gamma = X.astype(dtype), y.astype(dtype), gamma.astype(dtype)
        # prev_loss = np.mean(gamma)
        X, y = X.astype(dtype), y.astype(dtype)

        # 151 features are always 0.0 on MNIST 2 vs 6. And this number is even higher for smaller subsets of MNIST,
        # i.e. subsets of examples partitioned by tree splits.
        idx_non_trivial = np.abs(X).sum(axis=0) > 0.0
        features_to_check = np.random.permutation(np.where(idx_non_trivial)[0])[:self.n_trials_coord]

        n_coords = len(features_to_check)
        params, min_losses = np.zeros((n_coords, 4)), np.full(n_coords, np.inf)

        if parallel:
            n_proc = get_n_proc(n_ex)
            n_proc = min(n_coords, min(100, n_proc))
            batch_size = n_coords // n_proc
            n_batches = n_coords // batch_size + 1

            with ThreadPoolExecutor(max_workers=n_proc) as executor:
                procs = []
                for i_batch in range(n_batches):
                    coords = features_to_check[i_batch*batch_size:(i_batch+1)*batch_size]
                    args = (X, X[:, coords], y, gamma, model, eps, coords)
                    procs.append(executor.submit(self.fit_stump_batch, *args))

                # Process the results
                i_coord = 0
                for i_batch in range(n_batches):
                    res_many = procs[i_batch].result()
                    for res in res_many:
                        min_losses[i_coord], *params[i_coord, :] = res
                        i_coord += 1
        else:
            if order == -1 or order == 0:
                for i_coord, coord in enumerate(features_to_check):
                    if coord in coords_to_ignore:
                        continue
                    if order == -1:
                        min_losses[i_coord], *params[i_coord, :] = self.fit_stump(
                            X, X[:, coord], y, gamma, model, eps, coord)
                    # elif order > 0:
                    #     min_losses[i_coord], *params[i_coord, :] = self.fit_stump_Lp(
                    #         X, X[:, coord], y, gamma, model, eps, coord, order, precision = precision)
                    else:
                        min_losses[i_coord], *params[i_coord, :] = self.fit_stump_L0(
                            X, X[:, coord], y, eps, coord)
            else:
                threshold_cumu_value = self.build_cumu_threshold_value()
                gamma_same = self.certify_Lp_bound(X, y, eps[0] if isinstance(eps, tuple) else eps, threshold_cumu_value, (), 1 if isinstance(eps, tuple) else order , precision)
                for i_coord, coord in enumerate(features_to_check):
                    if coord in self.coords_trees:
                        gamma_general = self.certify_Lp_bound(X, y, eps[0] if isinstance(eps, tuple) else eps, threshold_cumu_value, (coord,), 1 if isinstance(eps, tuple) else order, precision)
                    else:
                        gamma_general = gamma_same
                    # gamma_same = self.certify_Lp_bound(X, y, eps, threshold_cumu_value, (coord,), order, precision)
                    if order > 0:
                        min_losses[i_coord], *params[i_coord, :] = self.fit_stump_Lp(
                                X, X[:, coord], y, gamma_general, model, eps, coord, order, precision = precision)
                    else:
                        min_losses[i_coord], *params[i_coord, :] = self.fit_stump_universal(\
                                X, X[:, coord], y, gamma_general, gamma, model, eps, coord, precision = precision)

        # print(min_losses)
        # print(params)
        id_best_coord = min_losses.argmin()
        min_loss = min_losses[id_best_coord]
        best_coord = int(params[id_best_coord][3])  # float to int is necessary for a coordinate
        best_wl, best_wr, best_b = params[id_best_coord][0], params[id_best_coord][1], np.float32(params[id_best_coord][2])
        if verbose:
            print('[{}-vs-all]: n_ex {}, n_coords {} -- loss {:.5f}, b={:.3f} wl={:.3f} wr={:.3f} at coord {}'.format(
                self.idx_clsf, n_ex, n_coords, min_loss, best_b, best_wl, best_wr, best_coord))
        return Stump(best_wl, best_wr, best_b, best_coord, min_loss)

    def fit_stump_batch(self, X, Xs, y, gamma, model, eps, coords):
        res = np.zeros([len(coords), 5])
        for i, coord in enumerate(coords):
            res[i] = self.fit_stump(X, Xs[:, i], y, gamma, model, eps, coord)
        return res

    def fit_stump_L0(self, X, X_proj, y, eps, coord):
        trees_current_coord = self.coords_trees[coord] if coord in self.coords_trees else []
        threshold_value ={}

        pre_y_min_1 = self.certify_exact_norm_zero(X, y, eps - 1, (coord,))
        pre_y_min_0 = self.certify_exact_norm_zero(X, y, eps, (coord,))

        # pre_y_min_1 = self.certify_exact_norm_zero(X, y, eps - 1, coord)
        # pre_y_min_0 = self.certify_exact_norm_zero(X, y, eps, coord)

        sum_w_l = 0
        for tree in trees_current_coord:
            sum_w_l += tree.w_l
            threshold_value[tree.b] = tree.w_r

        # print(threshold_value)
        sorted_thresholds = sorted(list(threshold_value.items()), key=lambda x: x[0])
        sorted_thresholds.insert(0, (-np.inf, 0))
        sorted_thresholds.append((np.inf, np.inf))

        interval_value = []
        pre_value = 0
        ori_value = np.zeros_like(y)
        # min_value = np.zeros_like(y) + np.inf
        for i, (b, value) in enumerate(sorted_thresholds[:-1]):
            pre_value += value
            interval_value.append((b, sorted_thresholds[i + 1][0], pre_value + sum_w_l))
            ori_value[b <= X_proj] = y[b <= X_proj] * pre_value
        #     min_value = np.minimum(min_value, y * pre_value)

        n_bins = self.n_bins
        b_vals = np.arange(0, n_bins) / n_bins
        # to have some margin to make the thresholds not adversarially reachable from 0 or 1
        b_vals[b_vals < 0.5] += 0.1 * 1 / n_bins
        b_vals[b_vals > 0.5] -= 0.1 * 1 / n_bins

        # print(b_vals)

        losses, w_l_vals, w_r_vals, b_vals = fit_robust_exact_stumps_L0(X_proj, y, pre_y_min_0, pre_y_min_1, interval_value, ori_value, b_vals, self.max_weight)

        # print(losses)
        
        min_loss = np.min(losses)
        # probably, they are already sorted, but to be 100% sure since it is not explicitly mentioned in the docs
        indices_opt_init = np.sort(np.where(losses == min_loss)[0])
        indices_opt = get_contiguous_indices(indices_opt_init)
        id_opt = indices_opt[len(indices_opt) // 2]

        # idx_prev = np.clip(indices_opt[0]-1, 0, len(b_vals)-1)  # to prevent stepping out of the array
        # idx_next = np.clip(indices_opt[-1]+1, 0, len(b_vals)-1)  # to prevent stepping out of the array
        # b_prev, w_l_prev, w_r_prev = b_vals[idx_prev], w_l_vals[idx_prev], w_r_vals[idx_prev]
        # b_next, w_l_next, w_r_next = b_vals[idx_next], w_l_vals[idx_next], w_r_vals[idx_next]
        # initialization
        b_leftmost, b_rightmost = b_vals[indices_opt[0]], b_vals[indices_opt[-1]]
        # more involved, since with +-eps, an additional check of the loss is needed

        # we put in the middle of the interval of the constant loss
        b_opt = (b_leftmost + b_rightmost) / 2

        # For the chosen threshold, we need to calculate w_l, w_r
        # Some of w_l, w_r that correspond to min_loss may not be optimal anymore
        b_val_final = np.array([b_opt])
        loss, w_l_opt, w_r_opt, _ = fit_robust_exact_stumps_L0(X_proj, y, pre_y_min_0, pre_y_min_1, interval_value, ori_value, b_val_final, self.max_weight)
        loss, w_l_opt, w_r_opt = loss[0], w_l_opt[0], w_r_opt[0]
        # recalculation of w_l, w_r shouldn't change the min loss

        if np.abs(loss - min_loss) > 1e7:
            print('New loss: {:.5f}, min loss before: {:.5f}'.format(loss, min_loss))

        best_loss = losses[id_opt]
        return [best_loss, w_l_opt, w_r_opt, b_opt, coord]

    
    def fit_stump_universal(self, X, X_proj, y, gamma_1, gamma_infty, model, eps, coord, precision = 0.001, lambda_1 = 0.7):
        assert isinstance(eps, tuple) and len(eps) == 2

        trees_current_coord = self.coords_trees[coord] if coord in self.coords_trees else []
        w_rs, bs = np.zeros(len(trees_current_coord)), np.zeros(len(trees_current_coord))

        for i in range((len(trees_current_coord))):
            w_rs[i] = trees_current_coord[i].w_r
            bs[i] = trees_current_coord[i].b

        cumu_threshold_value = self.build_cumu_threshold_value()

        if gamma_1 is None:
            gamma_1 = self.certify_Lp_bound(X, y, eps[0], cumu_threshold_value, (coord,), 1, precision)

        intervals_j = cumu_threshold_value[coord] if coord in cumu_threshold_value else [(-np.inf, np.inf, 0)]
        n_bins = self.n_bins

        b_vals_universal = np.arange(1, n_bins) / n_bins
        # to have some margin to make the thresholds not adversarially reachable from 0 or 1
        b_vals_universal[b_vals_universal < 0.5] += 0.1 * 1 / n_bins
        b_vals_universal[b_vals_universal > 0.5] -= 0.1 * 1 / n_bins

        intervals = tuple([(i[0], i[1]) for i in intervals_j])
        values = tuple([i[2] for i in intervals_j])
        # fit_robust_exact_stumps_universal(X_proj, y, y_min_without_j, threshold_cumu_value, gamma, b_vals, eps, w_rs, bs, max_weight, precision, verbose = False):
        losses_universal, w_l_vals_universal, w_r_vals_universal, b_vals_universal = fit_robust_exact_stumps_universal(X_proj, y, gamma_1, intervals, values, gamma_infty, b_vals_universal, eps, w_rs, bs, self.max_weight, precision, lambda_1, verbose = False)
        

        min_loss = np.min(losses_universal)
        # probably, they are already sorted, but to be 100% sure since it is not explicitly mentioned in the docs
        indices_opt_init = np.sort(np.where(losses_universal == min_loss)[0])
        indices_opt = get_contiguous_indices(indices_opt_init)

        id_opt = indices_opt[len(indices_opt) // 2]

        loss = losses_universal[id_opt]

        # if id_opt >= len(b_vals_1):
        # w_l, w_r, b = w_l_vals_infty[id_opt - len(b_vals_1)], w_r_vals_infty[id_opt - len(b_vals_1)], b_vals_infty[id_opt - len(b_vals_1)]
        # else:
        w_l, w_r, b = w_l_vals_universal[id_opt], w_r_vals_universal[id_opt], b_vals_universal[id_opt]

        if np.abs(loss - min_loss) > 1e7:
            print('New loss: {:.5f}, min loss before: {:.5f}'.format(loss, min_loss))

        # best_loss = losses[id_opt]
        # return [best_loss, w_l_opt, w_r_opt, b_opt, coord]

        return [loss, w_l, w_r, b, coord]




    def fit_stump_Lp(self, X, X_proj, y, gamma_global, model, eps, coord, order = 1, precision = 0.02):
        trees_current_coord = self.coords_trees[coord] if coord in self.coords_trees else []
        w_rs, bs = np.zeros(len(trees_current_coord)), np.zeros(len(trees_current_coord))

        for i in range((len(trees_current_coord))):
            w_rs[i] = trees_current_coord[i].w_r
            bs[i] = trees_current_coord[i].b

        cumu_threshold_value = self.build_cumu_threshold_value()
        # ori_y_without_feature_j = self.predict(X, (coord,))

        begin = time.time()
        if gamma_global is None:
            gamma_global = self.certify_Lp_bound(X, y, eps, cumu_threshold_value, (coord,), order, precision)
        # print("certification: {}".format(time.time() - begin))

        # print(y_min_without_feature_j)

        intervals_j = cumu_threshold_value[coord] if coord in cumu_threshold_value else [(-np.inf, np.inf, 0)]
        # print(coord, intervals_j)
        # sum_w_l = np.sum([tree.w_l for tree in trees_current_coord])
        # intervals_j = [(a, b, value) for (a, b, value) in intervals_j] # value of each interval(ignore the newly built stump)

        # print(intervals_j)
        
        n_bins = self.n_bins

        if eps < 0.5:
            b_vals = np.arange(eps * n_bins, n_bins - eps * n_bins + 1) / n_bins
        else:
            b_vals = np.arange(1, n_bins) / n_bins
        # to have some margin to make the thresholds not adversarially reachable from 0 or 1
        b_vals[b_vals < 0.5] += 0.1 * 1 / n_bins
        b_vals[b_vals > 0.5] -= 0.1 * 1 / n_bins

        # gamma = y_min_without_feature_j # y.shape[0] * C
        intervals = tuple([(i[0], i[1]) for i in intervals_j])
        values = tuple([i[2] for i in intervals_j])

        # print(intervals, values)

        begin = time.time()
        losses, w_l_vals, w_r_vals, b_vals = fit_robust_exact_stumps_Lp(X_proj, y, gamma_global, intervals, values, b_vals, eps, w_rs, bs, self.max_weight, order, precision)
        # print("fit stumps: {}".format(time.time() - begin))
        # print(losses, w_l_vals)

        min_loss = np.min(losses)
        # probably, they are already sorted, but to be 100% sure since it is not explicitly mentioned in the docs
        indices_opt_init = np.sort(np.where(losses == min_loss)[0])
        indices_opt = get_contiguous_indices(indices_opt_init)
        id_opt = indices_opt[len(indices_opt) // 2]

        # initialization
        b_leftmost, b_rightmost = b_vals[indices_opt[0]], b_vals[indices_opt[-1]]
        b_opt = (b_leftmost + b_rightmost) / 2

        # For the chosen threshold, we need to calculate w_l, w_r
        # Some of w_l, w_r that correspond to min_loss may not be optimal anymore
        b_val_final = np.array([b_opt])
        # print('-----------------{}-----------------'.format(coord))
        loss, w_l_opt, w_r_opt, _ = fit_robust_exact_stumps_Lp(X_proj, y, gamma_global, intervals, values, b_val_final, eps, w_rs, bs, self.max_weight, order, precision, verbose = False)
        # else:
        #     raise ValueError('wrong model')
        loss, w_l_opt, w_r_opt = loss[0], w_l_opt[0], w_r_opt[0]
        # recalculation of w_l, w_r shouldn't change the min loss

        if np.abs(loss - min_loss) > 1e7:
            print('New loss: {:.5f}, min loss before: {:.5f}'.format(loss, min_loss))

        best_loss = losses[id_opt]
        return [best_loss, w_l_opt, w_r_opt, b_opt, coord]

    def fit_stump(self, X, X_proj, y, gamma_global, model, eps, coord):
        min_prec_val = 1e-7
        min_val, max_val = 0.0, 1.0  # can be changed if the features are in a different range
        n_bins = self.n_bins

        # Needed for exact robust optimization with stumps
        trees_current_coord = self.coords_trees[coord] if coord in self.coords_trees else []
        w_rs, bs = np.zeros(len(trees_current_coord)), np.zeros(len(trees_current_coord))
        for i in range(len(trees_current_coord)):
            w_rs[i] = trees_current_coord[i].w_r
            bs[i] = trees_current_coord[i].b

        if model == 'robust_exact' and trees_current_coord != []:  # note: the previous gamma is just ignored
            min_Fx_y_exact_without_j = self.certify_exact(X, y, eps, coords_to_ignore=(coord, ))
            w_ls = np.sum([tree.w_l for tree in trees_current_coord])
            gamma = np.exp(-min_Fx_y_exact_without_j - y*w_ls)
        else:
            gamma = gamma_global

        if n_bins > 0:
            if model == 'robust_bound':
                # b_vals = np.array([0.31, 0.41, 0.5, 0.59, 0.69])  # that's the thresholds that one gets with n_bins=10
                b_vals = np.arange(eps * n_bins, n_bins - eps * n_bins + 1) / n_bins
                # to have some margin to make the thresholds not adversarially reachable from 0 or 1
                b_vals[b_vals < 0.5] += 0.1 * 1 / n_bins
                b_vals[b_vals > 0.5] -= 0.1 * 1 / n_bins
            else:
                b_vals = np.arange(1, n_bins) / n_bins
        else:
            threshold_candidates = np.sort(X_proj)
            if len(threshold_candidates) == 0:  # if no samples left according to min_samples_leaf
                return [np.inf, 0.0, 0.0, 0.0, -1]
            if model not in ['robust_bound', 'robust_exact'] or eps == 0.0:  # plain, da_uniform or at_cube training
                b_vals = np.copy(threshold_candidates)
                b_vals += min_prec_val  # to break the ties
            else:  # robust training
                b_vals = np.concatenate((threshold_candidates - eps, threshold_candidates + eps), axis=0)
                b_vals = np.clip(b_vals, min_val, max_val)  # save computations (often goes 512 -> 360 thresholds on MNIST)
                # to make in the overlapping case [---x-[--]-x---] output 2 different losses in the middle
                n_bs = len(threshold_candidates)
                b_vals += np.concatenate((-np.full(n_bs, min_prec_val), np.full(n_bs, min_prec_val)), axis=0)
            b_vals = np.unique(b_vals)  # use only unique b's
            b_vals = np.sort(b_vals)  # still important to sort because of the final threshold selection

        if model in ['plain', 'da_uniform', 'at_cube']:
            losses, w_l_vals, w_r_vals, b_vals = fit_plain_stumps(X_proj, y, gamma, b_vals, self.max_weight)
        elif model == 'robust_bound':
            losses, w_l_vals, w_r_vals, b_vals = fit_robust_bound_stumps(X_proj, y, gamma, b_vals, eps, self.max_weight)
        elif model == 'robust_exact':
            losses, w_l_vals, w_r_vals, b_vals = fit_robust_exact_stumps(X_proj, y, gamma, b_vals, eps, w_rs, bs, self.max_weight)
        else:
            raise ValueError('wrong model')

        min_loss = np.min(losses)
        # probably, they are already sorted, but to be 100% sure since it is not explicitly mentioned in the docs
        indices_opt_init = np.sort(np.where(losses == min_loss)[0])
        indices_opt = get_contiguous_indices(indices_opt_init)
        id_opt = indices_opt[len(indices_opt) // 2]

        idx_prev = np.clip(indices_opt[0]-1, 0, len(b_vals)-1)  # to prevent stepping out of the array
        idx_next = np.clip(indices_opt[-1]+1, 0, len(b_vals)-1)  # to prevent stepping out of the array
        b_prev, w_l_prev, w_r_prev = b_vals[idx_prev], w_l_vals[idx_prev], w_r_vals[idx_prev]
        b_next, w_l_next, w_r_next = b_vals[idx_next], w_l_vals[idx_next], w_r_vals[idx_next]
        # initialization
        b_leftmost, b_rightmost = b_vals[indices_opt[0]], b_vals[indices_opt[-1]]
        # more involved, since with +-eps, an additional check of the loss is needed
        if model in ['plain', 'da_uniform', 'at_cube']:
            b_rightmost = b_next
        elif model in ['robust_bound', 'robust_exact']:
            h_flag = False if model == 'robust_bound' else True

            b_prev_half = (b_prev + b_vals[indices_opt[0]]) / 2
            loss_prev_half = exp_loss_robust(X_proj, y, gamma, w_l_prev, w_r_prev, w_rs, bs, b_prev_half, eps, h_flag)

            b_next_half = (b_vals[indices_opt[-1]] + b_next) / 2
            loss_next_half = exp_loss_robust(X_proj, y, gamma, w_l_next, w_r_next, w_rs, bs, b_next_half, eps, h_flag)

            # we extend the interval of the constant loss to the left and to the right if there the loss is
            # the same at b_prev_half or b_next_half
            if loss_prev_half == losses[id_opt]:
                b_leftmost = b_prev
            if loss_next_half == losses[id_opt]:
                b_rightmost = b_next
        else:
            raise ValueError('wrong model')

        # we put in the middle of the interval of the constant loss
        b_opt = (b_leftmost + b_rightmost) / 2

        # For the chosen threshold, we need to calculate w_l, w_r
        # Some of w_l, w_r that correspond to min_loss may not be optimal anymore
        b_val_final = np.array([b_opt])
        if model in ['plain', 'da_uniform', 'at_cube']:
            loss, w_l_opt, w_r_opt, _ = fit_plain_stumps(X_proj, y, gamma, b_val_final, self.max_weight)
        elif model == 'robust_bound':
            loss, w_l_opt, w_r_opt, _ = fit_robust_bound_stumps(X_proj, y, gamma, b_val_final, eps, self.max_weight)
        elif model == 'robust_exact':
            loss, w_l_opt, w_r_opt, _ = fit_robust_exact_stumps(X_proj, y, gamma, b_val_final, eps, w_rs, bs, self.max_weight)
        else:
            raise ValueError('wrong model')
        loss, w_l_opt, w_r_opt = loss[0], w_l_opt[0], w_r_opt[0]
        # recalculation of w_l, w_r shouldn't change the min loss

        if np.abs(loss - min_loss) > 1e7:
            print('New loss: {:.5f}, min loss before: {:.5f}'.format(loss, min_loss))

        best_loss = losses[id_opt]
        return [best_loss, w_l_opt, w_r_opt, b_opt, coord]


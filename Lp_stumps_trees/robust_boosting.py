import numpy as np
import math
from numba import jit, prange
from utils import minimum, clip
import numba
import time

from box import Box


dtype = np.float32  # float32 is much faster than float64 because of exp
parallel = False  # and then it also depends on NUMBA_NUM_THREADS
nogil = True


@jit(nopython=True, nogil=nogil)
def fit_plain_stumps_iter(X_proj, y, gamma, b_vals_i, sum_1, sum_m1, max_weight):
    ind = X_proj >= b_vals_i
    sum_1_1, sum_1_m1 = np.sum(ind * (y == 1) * gamma), np.sum(ind * (y == -1) * gamma)
    sum_0_1, sum_0_m1 = sum_1 - sum_1_1, sum_m1 - sum_1_m1
    w_l, w_r = coord_descent_exp_loss(sum_1_1, sum_1_m1, sum_0_1, sum_0_m1, max_weight)

    fmargin = y * w_l + y * w_r * ind
    loss = np.mean(gamma * np.exp(-fmargin))
    return loss, w_l, w_r


@jit(nopython=True, nogil=nogil, parallel=parallel)  # really matters, especially with independent iterations
def fit_plain_stumps(X_proj, y, gamma, b_vals, max_weight):
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
    for i in prange(n_thresholds):
        # due to a numba bug, if we don't use a separate function inside a prange-loop, we experience a memory leak
        losses[i], w_l_vals[i], w_r_vals[i] = fit_plain_stumps_iter(
            X_proj, y, gamma, b_vals[i], sum_1, sum_m1, max_weight)
    return losses, w_l_vals, w_r_vals, b_vals


@jit(nopython=True, nogil=nogil)
def fit_robust_bound_stumps_iter(X_proj, y, gamma, b_vals_i, sum_1, sum_m1, eps, max_weight):
    # Certification for the previous ensemble O(n)
    split_lbs, split_ubs = X_proj - eps, X_proj + eps
    guaranteed_right = split_lbs > b_vals_i
    uncertain = (split_lbs <= b_vals_i) * (split_ubs >= b_vals_i)

    loss, w_l, w_r = basic_case_two_intervals(y, gamma, guaranteed_right, uncertain, sum_1, sum_m1, max_weight)
    return loss, w_l, w_r


@jit(nopython=True, nogil=nogil, parallel=parallel)  # parallel=True really matters, especially with independent iterations
def fit_robust_bound_stumps(X_proj, y, gamma, b_vals, eps, max_weight):
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
    for i in prange(n_thresholds):
        losses[i], w_l_vals[i], w_r_vals[i] = fit_robust_bound_stumps_iter(
            X_proj, y, gamma, b_vals[i], sum_1, sum_m1, eps, max_weight)

    return losses, w_l_vals, w_r_vals, b_vals


@jit(nopython=True, nogil=nogil)
def fit_robust_exact_stumps_iter(X_proj, y, gamma, w_rs, bs, b_vals_i, sum_1, sum_m1, eps, max_weight):
    # Certification for the previous ensemble O(n)
    split_lbs, split_ubs = X_proj - eps, X_proj + eps
    guaranteed_right = split_lbs > b_vals_i
    uncertain = (split_lbs <= b_vals_i) * (split_ubs >= b_vals_i)

    h_l, h_r = calc_h(X_proj, y, w_rs, bs, b_vals_i, eps)
    # there should be quite many useless coordinates which do not have any stumps in the ensemble
    # thus h_l=h_r=0  =>  suffices to check just 2 regions without applying bisection
    if np.sum(h_l) == 0.0 and np.sum(h_r) == 0.0:
        loss, w_l, w_r = basic_case_two_intervals(y, gamma, guaranteed_right, uncertain, sum_1, sum_m1, max_weight)
    else:  # general case; happens only when `coord` was already splitted in the previous iterations
        loss, w_l, w_r = bisect_coord_descent(y, gamma, h_l, h_r, guaranteed_right, uncertain, max_weight)
    return loss, w_l, w_r


@jit(nopython=True, nogil=nogil)
def calc_h_Lp(X_proj, y, y_min_without_j, intervals, values, b_vals_i, order, precision, C, verbose = False):
    '''
    for illegal position, h = np.inf
    '''
    num = X_proj.shape[0]
    # threshold_value_j = zip(intervals, values)

    h_l, h_r = np.zeros(num) + np.inf, np.zeros(num) + np.inf # h_l is the min_score when at the left, h_r is the min_score when at the right(ignore the newly added stump)


    for i in range(num):
        x_i = X_proj[i]
        y_i = int(y[i])

        pre_value = y_min_without_j[i][C]

        # threshold_value_j_times = [(a, b, y_i * value) for (a, b, value) in threshold_value_j]
        for (a, b), value in zip(intervals, values):
            value *= y_i
            if b < x_i:
                gap = int(abs((x_i - b) ** order / precision))
            elif a > x_i:
                gap = int(abs((x_i - a) ** order / precision))
            elif b == x_i:
                gap = 1
            elif a == x_i:
                gap = 0
            else: # x_i \in (a, b) # TODO: Yihan: bug in boundary cases
                if a >= b_vals_i:
                    h_r[i] = min(h_r[i], pre_value + value)
                elif b <= b_vals_i:
                    h_l[i] = min(h_l[i], pre_value + value)
                else:
                    gap = int(abs((x_i - b_vals_i) ** order / precision))
                    if x_i < b_vals_i:
                        h_l[i] = min(h_l[i], pre_value + value)
                        if gap <= C:
                            h_r[i] = min(h_r[i], y_min_without_j[i][C - gap] + value)
                    else:
                        h_r[i] = min(h_r[i], pre_value + value)
                        if gap <= C:
                            h_l[i] = min(h_l[i], y_min_without_j[i][C - gap] + value)
                continue
            if b <= b_vals_i and C >= gap:
                h_l[i] = min(h_l[i], y_min_without_j[i][C - gap] + value)
            elif a >= b_vals_i and C >= gap:
                h_r[i] = min(h_r[i], y_min_without_j[i][C - gap] + value)
            # elif b == b_vals_i and C >= gap:
                # h_l[i] = min(h_l[i], y_min_without_j[i][C - gap] + value)
            # elif a == b_vals_i and C >= gap:
                # h_l[i] = min(h_l[i], y_min_without_j[i][C - gap] + value)
            else:
                if b <= x_i:
                    if C >= gap:
                        h_r[i] = min(h_r[i], y_min_without_j[i][C - gap] + value)
                    gap = int(abs((x_i - b_vals_i) ** order / precision))
                    if C >= gap:
                        h_l[i] = min(h_l[i], y_min_without_j[i][C - gap] + value)
                elif a > x_i:
                    if C >= gap:
                        h_l[i] = min(h_l[i], y_min_without_j[i][C - gap] + value)
                    gap = int(abs((x_i - b_vals_i) ** order / precision))
                    if C >= gap:
                        h_r[i] = min(h_r[i], y_min_without_j[i][C - gap] + value)

    # print(h_l, h_r, b_vals_i)
    return h_l, h_r

@jit(nopython=True, nogil=nogil)
def fit_robust_exact_stumps_universal_iter(X_proj, y, y_min_without_j, intervals, values, gamma, w_rs, bs, b_vals_i, eps, max_weight, precision, lambda_1, verbose = False):
    C = math.ceil(eps[0]/precision)
    
    if len(intervals) == 1 and np.sum(y_min_without_j) == 0.0:
        # print('--')
        split_lbs, split_ubs = X_proj - eps[0], X_proj + eps[0]
        guaranteed_right = split_lbs > b_vals_i
        uncertain = (split_lbs <= b_vals_i) * (split_ubs >= b_vals_i)

        loss, w_l, w_r = basic_case_two_intervals(y, np.zeros(y.shape) + 1, guaranteed_right, uncertain, np.sum(y == 1), np.sum(y == -1), max_weight)
    else:
        h_l_1, h_r_1 = calc_h_Lp(X_proj, y, y_min_without_j, intervals, values, b_vals_i, 1, precision, C)
        h_l_infty, h_r_infty = calc_h(X_proj, y, w_rs, bs, b_vals_i, eps[1])

        split_lbs, split_ubs = X_proj - eps[1], X_proj + eps[1]
        guaranteed_right = split_lbs > b_vals_i
        uncertain = (split_lbs <= b_vals_i) * (split_ubs >= b_vals_i)


        # TODO: here we should optimize these two forms simultaneuosly
        # bisect_coord_descent_universal(y, gamma, h_l_1, h_r_1, h_l_infty, h_r_infty, guaranteed_right, uncertain, max_weight, lambda_1 = 0.7, verbose= False):
        loss, w_l, w_r = bisect_coord_descent_universal(y, gamma, h_l_1, h_r_1, h_l_infty, h_r_infty,  guaranteed_right, uncertain, max_weight, lambda_1, verbose = verbose)

    return loss, w_l, w_r

@jit(nopython=True, nogil=nogil)
def fit_robust_exact_stumps_Lp_iter(X_proj, y, y_min_without_j, intervals, values, w_rs, bs, b_vals_i, eps, max_weight, order, precision, verbose = False):
    # Certification for the previous ensemble O(n)
    C = math.ceil(eps ** order/precision)
    # h_l, h_r = calc_h_Lp(X_proj, y, y_min_without_j, threshold_value_j, b_vals_i, order, precision, C)

    # print(h_l, h_r)

    # print(np.sum(y_min_without_j), )
    if len(intervals) == 1 and np.sum(y_min_without_j) == 0.0:
        # print('--')
        split_lbs, split_ubs = X_proj - math.pow(eps, 1/order), X_proj + math.pow(eps, 1/order)
        guaranteed_right = split_lbs > b_vals_i
        uncertain = (split_lbs <= b_vals_i) * (split_ubs >= b_vals_i)

        loss, w_l, w_r = basic_case_two_intervals(y, np.zeros(y.shape) + 1, guaranteed_right, uncertain, np.sum(y == 1), np.sum(y == -1), max_weight)
    else:
        # h_l, h_r = np.zeros((y.shape[0])), np.zeros((y.shape[0]))
        # begin = time.time()
        h_l, h_r = calc_h_Lp(X_proj, y, y_min_without_j, intervals, values, b_vals_i, order, precision, C)
        # print(h_l, h_r)
        # print("calculate hs: {}".format(time.time() - begin))
        # print(numba.typeof(h_r))

        # begin = time.time()
        loss, w_l, w_r = bisect_coord_descent_Lp(y, h_l, h_r, max_weight, verbose)
        # print("bisect: {}".format(time.time() - begin))

    # print(loss)

    return loss, w_l, w_r

def fit_robust_exact_stumps_L0(X_proj, y, pre_y_min_0, pre_y_min_1, interval_value, ori_value, b_vals, max_weight):

    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)

    no_perturb = pre_y_min_0 + y * ori_value

    for i in range(n_thresholds):
        h_l = np.zeros_like(y) + np.inf
        h_r = np.zeros_like(y) + np.inf

        b_i = b_vals[i]

        h_l[X_proj < b_i] = no_perturb[X_proj < b_i]
        h_r[X_proj >= b_i] = no_perturb[X_proj >= b_i]

        # print(interval_value)
        for a, b, value in interval_value:
            if b <= b_i:
                h_l = np.minimum(h_l, pre_y_min_1 + y * value)
            elif a > b_i:
                h_r = np.minimum(h_r, pre_y_min_1 + y * value)
            else:
                h_l = np.minimum(h_l, pre_y_min_1 + y * value)
                h_r = np.minimum(h_r, pre_y_min_1 + y * value)
        
        eps_precision = 1e-5
        w_l_prev, w_r_prev = np.inf, np.inf
        w_l, w_r = 0.0, 0.0

        i = 0
        while np.abs(w_l - w_l_prev) > eps_precision or np.abs(w_r - w_r_prev) > eps_precision:
            w_r_prev = w_r
            w_r, gamma, ind_left, ind_right = bisection_L0(w_l, y, h_l, h_r, max_weight)

            sum_1_1, sum_1_m1 = np.sum(gamma * ind_right * (y == 1)), np.sum(gamma * ind_right * (y == -1))
            sum_0_1, sum_0_m1 = np.sum(gamma * ind_left * (y == 1)), np.sum(gamma * ind_left * (y == -1))

            w_l_prev = w_l

            # print(h_r)

            if sum_1_m1 == 0 and sum_0_m1 == 0:
                w_l = max_weight * math.copysign(1, 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1)))
            else:
                # print(sum_1_1, sum_0_1, sum_1_m1, sum_0_m1)
                w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))

            # print(w_l, w_r)

        gamma[ind_left] = np.exp(-(h_l[ind_left] + y[ind_left] * w_l))
        gamma[ind_right] = np.exp(-(h_r[ind_right] + y[ind_right] * (w_r + w_l)))

        loss = np.mean(gamma)

        losses[i], w_l_vals[i], w_r_vals[i] = loss, w_l, w_r

    return losses, w_l_vals, w_r_vals, b_vals

def fit_robust_bound_stumps_tree_Lp(X_index, X, X_proj, y, budget, coord, leaf_nodes, b_vals, eps, max_eps, order, max_weight, curr_box):
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)

    for i in prange(n_thresholds):

        h_l = np.zeros_like(y) + np.inf
        h_r = np.zeros_like(y) + np.inf
        # here we calculate the new gamma.
        # fit a new L-infity stump
        
        box_left = curr_box.get_intersection(Box({coord: (-np.inf, b_vals[i])}))
        box_right = curr_box.get_intersection(Box({coord: (b_vals[i], np.inf)}))
        # box_right = 

        # each lead node should contain a set {idx: y_min_pre on this node}
        for leaf in leaf_nodes:
            intersected_box_left, intersected_box_right = None, None
            # here we calculate the minimized score in each box, and select the boxes according to L_L + y_i w_l and L_R + y_i(w_r + w_l).
            # but when we build the first tree, there is no 
            if box_left.is_intersect(leaf.box):
                intersected_box_left = leaf.box.get_intersection(box_left) # 取交集结束之后的左边交叉盒
                # for i in index_left:

                # pass
            if box_right.is_intersect(leaf.box):
                intersected_box_right = leaf.box.get_intersection(box_right)

            for ii, (index, x_i, x_proj_i) in enumerate(zip(X_index, X, X_proj)):
                # print(list(leaf.y_min_value_map.keys()))
                # intervals = {coord:(intersected_box_left.intervals[coord].lower_bound, intersected_box_left.intervals[coord].upper_bound) for coord in intersected_box_left.intervals}
                if intersected_box_left is not None and point_box_dist(x_i, list(intersected_box_left.intervals.keys()), list(intersected_box_left.intervals.values()), order) < eps:
                    # if index not in leaf.y_min_value_map:
                    #     print(index)
                    # else:
                    if index in leaf.y_min_value_map:
                        h_l[ii] = min(h_l[ii], leaf.y_min_value_map[index])

                # intervals = {coord:(intersected_box_right.intervals[coord].lower_bound, intersected_box_right.intervals[coord].upper_bound) for coord in intersected_box_right.intervals}
                if intersected_box_right is not None and point_box_dist(x_i, list(intersected_box_right.intervals.keys()), list(intersected_box_right.intervals.values()), order) < eps:
                    # if index not in leaf.y_min_value_map:
                    #     print(index)
                    # else:
                    if index in leaf.y_min_value_map:
                        h_r[ii] = min(h_r[ii], leaf.y_min_value_map[index])
                # test whether intersect
        
        # print(i)
        losses[i], w_l_vals[i], w_r_vals[i] = bisect_coord_descent_Lp(y, h_l, h_r, max_weight)

    return losses, w_l_vals, w_r_vals, b_vals

@jit(nopython=True, nogil=nogil)
def point_interval_dist(value, l, u, order):
    dist = 0
    if value > u:
        if order > 0:
            dist = pow(value - u, order)
        elif order == 0:
            dist = 1
        else:
            dist = value - u
    elif value < l:
        if order > 0:
            dist = pow(l - value, order)
        elif order == 0:
            dist = 1
        else:
            dist = l - value
    return dist

@jit(nopython=True, nogil=nogil)
def point_box_dist(X_i, coords, intervals, order):
    res = 0
    # dist = 0

    for coord, interval in zip(coords, intervals):
        dist = point_interval_dist(X_i[coord], interval[0], interval[1], order)
    if order >= 0:
        res += dist
    else:
        res = max(res, dist)
    
    if order >= 0:
        return pow(res, 1/order)
    else:
        return res

@jit(nopython=True, nogil=nogil)
def fit_robust_exact_stumps_universal(X_proj, y, y_min_without_j, intervals, values, gamma, b_vals, eps, w_rs, bs, max_weight, precision, lambda_1, verbose = False):
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)

    for i in prange(n_thresholds):
        # print(X_proj, y, y_min_without_j, threshold_value_j, gamma, w_rs, bs, b_vals[i], eps, max_weight, precision)
        losses[i], w_l_vals[i], w_r_vals[i] = fit_robust_exact_stumps_universal_iter(X_proj, y, y_min_without_j, intervals, values, gamma, w_rs, bs, b_vals[i], eps, max_weight, precision, lambda_1, verbose = False)
        #fit_robust_exact_stumps_universal_iter(
            # X_proj, y, y_min_without_j, threshold_value_j, w_rs, bs, b_vals[i], eps, max_weight, order, precision, verbose = verbose)

    return losses, w_l_vals, w_r_vals, b_vals


@jit(nopython=True, nogil=nogil)
def fit_robust_exact_stumps_Lp(X_proj, y, y_min_without_j, intervals, values, b_vals, eps, w_rs, bs, max_weight, order, precision, verbose = False):
    '''
    X_proj: projection of X on coord j
    y_min_without_j: (num, C + 1) numpy array, min score ignoring feature j
    threshold_value_j: ignoring the newly built stump
    '''
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    # sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
    # print(b_vals)
    for i in prange(n_thresholds):
        losses[i], w_l_vals[i], w_r_vals[i] = fit_robust_exact_stumps_Lp_iter(
            X_proj, y, y_min_without_j, intervals, values, w_rs, bs, b_vals[i], eps, max_weight, order, precision, verbose = verbose)

    return losses, w_l_vals, w_r_vals, b_vals

@jit(nopython=True, nogil=nogil, parallel=parallel)  # parallel=True really matters, especially with independent iterations
def fit_robust_exact_stumps(X_proj, y, gamma, b_vals, eps, w_rs, bs, max_weight):
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
    for i in prange(n_thresholds):
        losses[i], w_l_vals[i], w_r_vals[i] = fit_robust_exact_stumps_iter(
            X_proj, y, gamma, w_rs, bs, b_vals[i], sum_1, sum_m1, eps, max_weight)

    return losses, w_l_vals, w_r_vals, b_vals


@jit(nopython=True, nogil=nogil)  # almost 2 times speed-up by njit for this loop!
def coord_descent_exp_loss(sum_1_1, sum_1_m1, sum_0_1, sum_0_m1, max_weight):
    m = 1e-10
    # if sum_0_1 + sum_0_m1 == 0 or sum_1_1 + sum_1_m1 == 0:
    #     return np.inf, np.inf
    # w_l = (sum_0_1 - sum_0_m1) / (sum_0_1 + sum_0_m1)
    # w_r = (sum_1_1 - sum_1_m1) / (sum_1_1 + sum_1_m1) - w_l

    # 1e-4 up to 20-50 iters; 1e-6 up to 100-200 iters which leads to a significant slowdown in practice
    eps_precision = 1e-4

    # We have to properly handle the cases when the optimal leaf value is +-inf.
    if sum_1_m1 < m and sum_0_1 < m:
        w_l, w_r = -max_weight, 2 * max_weight
    elif sum_1_1 < m and sum_0_m1 < m:
        w_l, w_r = max_weight, -2 * max_weight
    elif sum_1_m1 < m:
        w_r = max_weight
        w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
    elif sum_1_1 < m:
        w_r = -max_weight
        w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
    elif sum_0_1 < m:
        w_l = -max_weight
        w_r = 0.5 * math.log(sum_1_1 / sum_1_m1) - w_l
    elif sum_0_m1 < m:
        w_l = max_weight
        w_r = 0.5 * math.log(sum_1_1 / sum_1_m1) - w_l
    else:  # main case
        w_r = 0.0
        w_l = 0.0
        w_r_prev, w_l_prev = np.inf, np.inf
        i = 0
        # Note: ideally one has to calculate the loss, but O(n) factor would slow down everything here
        while (np.abs(w_r - w_r_prev) > eps_precision) or (np.abs(w_l - w_l_prev) > eps_precision):
            i += 1
            w_r_prev, w_l_prev = w_r, w_l
            w_r = 0.5 * math.log(sum_1_1 / sum_1_m1) - w_l
            w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
            if i == 50:
                break
    left_leaf = clip(w_l, -max_weight, max_weight)
    right_leaf = clip(left_leaf + w_r, -max_weight, max_weight)
    w_l, w_r = left_leaf, right_leaf - left_leaf
    return w_l, w_r


@jit(nopython=True, nogil=nogil)
def calc_h(X_proj, y, w_rs, bs, b_curr, eps):
    num = X_proj.shape[0]
    h_l_base, h_r_base = np.zeros(num), np.zeros(num)
    if len(bs) == 0:
        return h_l_base, h_r_base

    # Has to be calculated inside of the loop since depends on the current b
    for i in range(len(w_rs)):
        # idea: accumulate all the thresholds that preceed the leftmost point
        h_l_base += y * w_rs[i] * (X_proj - eps >= bs[i])  # leftmost point is `X_proj - eps`
        h_r_base += y * w_rs[i] * (np.maximum(b_curr, X_proj - eps) >= bs[i])  # leftmost point is max(b_curr, x-eps)
    # check all thresholds, and afterwards check if they are in (x-eps, x+eps]
    idx = np.argsort(bs)
    sorted_thresholds = bs[idx]
    sorted_w_r = w_rs[idx]

    min_left, min_right = np.zeros(num), np.zeros(num)
    cumsum_left, cumsum_right = np.zeros(num), np.zeros(num)
    for i_t in range(len(sorted_thresholds)):
        # consider the threshold if it belongs to (x-eps, min(b, x+eps)] (x-eps is excluded since already evaluated)
        idx_x_left = (X_proj - eps < sorted_thresholds[i_t]) * (sorted_thresholds[i_t] <= b_curr) * (
                sorted_thresholds[i_t] <= X_proj + eps)
        # consider the threshold if it belongs to (max(b, x-eps), x+eps] (b is excluded since already evaluated)
        idx_x_right = (b_curr < sorted_thresholds[i_t]) * (X_proj - eps < sorted_thresholds[i_t]) * (
                sorted_thresholds[i_t] <= X_proj + eps)
        assert np.sum(idx_x_left * idx_x_right) == 0  # mutually exclusive  =>  cannot be True at the same time
        diff_left = y * sorted_w_r[i_t] * idx_x_left
        diff_right = y * sorted_w_r[i_t] * idx_x_right
        # Note: numba doesn't support cumsum over axis=1 nor min over axis=1
        cumsum_left += diff_left
        cumsum_right += diff_right
        min_left = minimum(cumsum_left, min_left)
        min_right = minimum(cumsum_right, min_right)
    h_l = h_l_base + min_left
    h_r = h_r_base + min_right
    # That was the case when b is in [x-eps, x+eps]. If not, then:
    h_l = h_l * (b_curr >= X_proj - eps)  # zero out if b_curr < X_proj - eps
    h_r = h_r * (b_curr <= X_proj + eps)  # zero out if b_curr > X_proj + eps
    return h_l, h_r


@jit(nopython=True, nogil=nogil)
def bisection(w_l, y, gamma, h_l, h_r, guaranteed_right, uncertain, max_weight):
    # bisection to find w_r* for the current w_l
    eps_precision = 1e-5  # 1e-5: 21 steps, 1e-4: 18 steps (assuming max_weight=10)
    w_r = 0.0
    w_r_lower, w_r_upper = -max_weight, max_weight
    loss_best = np.inf
    i = 0
    while i == 0 or np.abs(w_r_upper - w_r_lower) > eps_precision:
        w_r = (w_r_lower + w_r_upper) / 2
        ind = guaranteed_right + (y * w_r < h_l - h_r) * uncertain

        # Calculate the indicator function based on the known h_l - h_r
        fmargin = y * w_l + h_l + (h_r - h_l + y * w_r) * ind
        losses_per_pt = gamma * np.exp(-fmargin)
        loss = np.mean(losses_per_pt)  # also O(n)
        # derivative wrt w_r for bisection
        derivative = np.mean(-losses_per_pt * y * ind)

        if loss < loss_best:
            w_r_best, loss_best = w_r, loss
        if derivative >= 0:
            w_r_upper = w_r
        else:
            w_r_lower = w_r

        i += 1
    return w_r


def bisection_L0(w_l, y, h_l, h_r, max_weight):
    eps_precision = 1e-5
    w_r = 0.0
    w_r_lower, w_r_upper = -max_weight, max_weight
    ind_left, ind_right = np.zeros(y.shape[0], dtype=np.bool_), np.zeros(y.shape, dtype=np.bool_)
    gamma = np.zeros(y.shape[0])

    loss_best = np.inf
    i = 0

    while i == 0 or np.abs(w_r_upper - w_r_lower) > eps_precision:
        w_r = (w_r_upper + w_r_lower) / 2

        # from h_l and h_r we calculate robust loss.
        ind_left = h_l < (h_r + y * w_r)
        ind_right = h_l >= (h_r + y * w_r)
# )
        gamma[ind_left] = np.exp(-(h_l[ind_left] + y[ind_left] * w_l))
        gamma[ind_right] = np.exp(-(h_r[ind_right] + y[ind_right] * (w_r + w_l)))


        loss = np.mean(gamma)
        derivative = np.mean(-gamma * y * ind_right)
        # print(derivative)

        if loss < loss_best:
            w_r_best, loss_best = w_r, loss
        if derivative >= 0:
            w_r_upper = w_r
        else:
            w_r_lower = w_r
        i += 1
    
    ind_left = h_l < (h_r + y * w_r)
    ind_right = h_l >= (h_r + y * w_r)
# )
    gamma[ind_left] = np.exp(-(h_l[ind_left]))
    gamma[ind_right] = np.exp(-(h_r[ind_right]))

    return w_r, gamma, ind_left, ind_right

@jit(nopython=True, nogil=nogil)
def bisection_universal(w_l, y, h_l_1, h_r_1, h_l_infty, h_r_infty, gamma, guaranteed_right, uncertain, max_weight, lambda_1):
    eps_precision = 1e-5
    w_r = 0.0
    w_r_lower, w_r_upper = -max_weight, max_weight

    gamma_1 = np.zeros(y.shape[0])
    ind_left, ind_right = np.zeros(y.shape[0], dtype=np.bool_), np.zeros(y.shape, dtype=np.bool_)
    loss_best = np.inf
    i = 0

    while i == 0 or np.abs(w_r_upper - w_r_lower) > eps_precision:
        w_r = (w_r_upper + w_r_lower) / 2

        # from h_l and h_r we calculate robust loss.
        ind_left = h_l_1 < (h_r_1 + y * w_r)
        ind_right = h_l_1 >= (h_r_1 + y * w_r)
# )
        gamma_1[ind_left] = np.exp(-(h_l_1[ind_left] + y[ind_left] * w_l))
        gamma_1[ind_right] = np.exp(-(h_r_1[ind_right] + y[ind_right] * (w_r + w_l)))


        loss = lambda_1 * np.mean(gamma_1)
        derivative = lambda_1 * np.mean(-gamma_1 * y * ind_right)

        ind = guaranteed_right + (y * w_r < h_l_infty - h_r_infty) * uncertain

        # Calculate the indicator function based on the known h_l - h_r
        fmargin = y * w_l + h_l_infty + (h_r_infty - h_l_infty + y * w_r) * ind
        losses_per_pt = gamma * np.exp(-fmargin)
        loss += (1 - lambda_1) * np.mean(losses_per_pt)  # also O(n)
        # derivative wrt w_r for bisection
        derivative += (1 - lambda_1) * np.mean(-losses_per_pt * y * ind)
        # print(derivative)

        if loss < loss_best:
            w_r_best, loss_best = w_r, loss
        if derivative >= 0:
            w_r_upper = w_r
        else:
            w_r_lower = w_r
        i += 1

    return w_r, gamma_1, ind_left, ind_right


@jit(nopython=True, nogil=nogil)
def bisection_Lp(w_l, y, h_l, h_r, max_weight):
    eps_precision = 1e-5
    w_r = 0.0
    w_r_lower, w_r_upper = -max_weight, max_weight
    gamma = np.zeros(y.shape[0])
    ind_left, ind_right = np.zeros(y.shape[0], dtype=np.bool_), np.zeros(y.shape, dtype=np.bool_)
    loss_best = np.inf
    i = 0

    # print(np.sum(h_l == np.inf))
    while i == 0 or np.abs(w_r_upper - w_r_lower) > eps_precision:
        w_r = (w_r_upper + w_r_lower) / 2

        # from h_l and h_r we calculate robust loss.
        ind_left = h_l < (h_r + y * w_r)
        ind_right = h_l >= (h_r + y * w_r)
# )
        gamma[ind_left] = np.exp(-(h_l[ind_left] + y[ind_left] * w_l))
        gamma[ind_right] = np.exp(-(h_r[ind_right] + y[ind_right] * (w_r + w_l)))


        loss = np.mean(gamma)
        derivative = np.mean(-gamma * y * ind_right)
        # print(derivative)

        if loss < loss_best:
            w_r_best, loss_best = w_r, loss
        if derivative >= 0:
            w_r_upper = w_r
        else:
            w_r_lower = w_r
        i += 1
    
    ind_left = h_l < (h_r + y * w_r)
    ind_right = h_l >= (h_r + y * w_r)
# )
    gamma[ind_left] = np.exp(-(h_l[ind_left]))
    gamma[ind_right] = np.exp(-(h_r[ind_right]))
    return w_r, gamma, ind_left, ind_right


@jit(nopython=True, nogil=nogil)
def bisect_coord_descent_universal(y, gamma, h_l_1, h_r_1, h_l_infty, h_r_infty, guaranteed_right, uncertain, max_weight = 1.0, lambda_1 = 0.3, verbose= False):
    eps_precision = 1e-5
    w_l_prev, w_r_prev = np.inf, np.inf
    w_l, w_r = 0.0, 0.0

    i = 0
    while np.abs(w_l - w_l_prev) > eps_precision or np.abs(w_r - w_r_prev) > eps_precision:
        w_r_prev = w_r
        # bisection_universal(w_l, y, h_l_1, h_r_1, h_l_infty, h_r_infty, gamma, guaranteed_right, uncertain, max_weight, lambda_1 = 0.7):
        w_r, gamma_1, ind_left, ind_right = bisection_universal(w_l, y, h_l_1, h_r_1, h_l_infty, h_r_infty, gamma, guaranteed_right, uncertain, max_weight, lambda_1)

        ind = guaranteed_right + (y * w_r < h_l_infty - h_r_infty) * uncertain
        gamma_with_h = gamma * np.exp(-(~ind * h_l_infty + ind * h_r_infty))  # only for the coord descent step
        sum_1_1, sum_1_m1 = (1 - lambda_1) * np.sum(ind * (y == 1) * gamma_with_h), (1 - lambda_1) * np.sum(ind * (y == -1) * gamma_with_h)
        sum_0_1, sum_0_m1 = (1 - lambda_1) * np.sum(~ind * (y == 1) * gamma_with_h), (1 - lambda_1) * np.sum(~ind * (y == -1) * gamma_with_h)

        sum_1_1 += lambda_1 * np.sum(gamma_1 * ind_right * (y == 1))
        sum_1_m1 += lambda_1 * np.sum(gamma_1 * ind_right * (y == -1))
        sum_0_1 += lambda_1 * np.sum(gamma_1 * ind_left * (y == 1))
        sum_0_m1 += lambda_1 * np.sum(gamma_1 * ind_left * (y == -1))

        w_l_prev = w_l

        if sum_1_m1 == 0 and sum_0_m1 == 0:
            w_l = max_weight * math.copysign(1, 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1)))
        else:
            w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
        # print(w_l)

        # print(w_r)
        # the same as Linf case, update w_l
        i += 1
        if i == 10:
            break

    if verbose:
        # pass
        print(h_l_1, h_r_1, h_l_infty, h_r_infty)
    # print(h_l, h_r)
    ind_left = h_l_1 < (h_r_1 + y * w_r)
    ind_right = h_l_1 >= (h_r_1 + y * w_r)

    # if verbose:
    #     y_min_score = np.zeros(h_l_1.shape[0])
    #     y_min_score[ind_left] = h_l[ind_left] + y[ind_left] * w_l
    #     y_min_score[ind_right] = h_r[ind_right] + y[ind_right] * (w_r + w_l)

    #     print("----------------------------------")
    #     print(y_min_score)

    gamma_1[ind_left] = np.exp(-(h_l_1[ind_left] + y[ind_left] * w_l))
    gamma_1[ind_right] = np.exp(-(h_r_1[ind_right] + y[ind_right] * (w_r + w_l)))

    loss = lambda_1 * np.mean(gamma_1)

    ind = guaranteed_right + (y * w_r < h_l_infty - h_r_infty) * uncertain
    fmargin = y * w_l + h_l_infty + (h_r_infty - h_l_infty + y * w_r) * ind
    loss += (1 - lambda_1) * np.mean(gamma * np.exp(-fmargin))

    return loss, w_l, w_r


@jit(nopython=True, nogil=nogil)
def bisect_coord_descent_Lp(y, h_l, h_r, max_weight, verbose = False):
    
    eps_precision = 1e-5
    w_l_prev, w_r_prev = np.inf, np.inf
    w_l, w_r = 0.0, 0.0

    i = 0
    while np.abs(w_l - w_l_prev) > eps_precision or np.abs(w_r - w_r_prev) > eps_precision:
        w_r_prev = w_r
        w_r, gamma, ind_left, ind_right = bisection_Lp(w_l, y, h_l, h_r, max_weight)


        sum_1_1, sum_1_m1 = np.sum(gamma * ind_right * (y == 1)), np.sum(gamma * ind_right * (y == -1))
        sum_0_1, sum_0_m1 = np.sum(gamma * ind_left * (y == 1)), np.sum(gamma * ind_left * (y == -1))

        w_l_prev = w_l

        if sum_1_m1 == 0 and sum_0_m1 == 0:
            w_l = max_weight * math.copysign(1, 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1)))
        else:
            w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
        # print(w_l)

        # print(w_r)
        # the same as Linf case, update w_l
        i += 1
        if i == 10:
            break

    if verbose:
        # pass
        print(h_l, h_r)
    # print(h_l, h_r)
    ind_left = h_l < (h_r + y * w_r)
    ind_right = h_l >= (h_r + y * w_r)

    if verbose:
        y_min_score = np.zeros(h_l.shape[0])
        y_min_score[ind_left] = h_l[ind_left] + y[ind_left] * w_l
        y_min_score[ind_right] = h_r[ind_right] + y[ind_right] * (w_r + w_l)

        print("----------------------------------")
        print(y_min_score)

    gamma[ind_left] = np.exp(-(h_l[ind_left] + y[ind_left] * w_l))
    gamma[ind_right] = np.exp(-(h_r[ind_right] + y[ind_right] * (w_r + w_l)))

    # print("gamma", gamma)
    # print(ind_)

    loss = np.mean(gamma)

    return loss, w_l, w_r

@jit(nopython=True, nogil=nogil)
def bisect_coord_descent(y, gamma, h_l, h_r, guaranteed_right, uncertain, max_weight):
    eps_precision = 1e-5
    w_l_prev, w_r_prev = np.inf, np.inf
    w_l, w_r = 0.0, 0.0
    i = 0
    while np.abs(w_l - w_l_prev) > eps_precision or np.abs(w_r - w_r_prev) > eps_precision:
        w_r_prev = w_r
        w_r = bisection(w_l, y, gamma, h_l, h_r, guaranteed_right, uncertain, max_weight)

        ind = guaranteed_right + (y * w_r < h_l - h_r) * uncertain
        gamma_with_h = gamma * np.exp(-(~ind * h_l + ind * h_r))  # only for the coord descent step
        sum_1_1, sum_1_m1 = np.sum(ind * (y == 1) * gamma_with_h), np.sum(ind * (y == -1) * gamma_with_h)
        sum_0_1, sum_0_m1 = np.sum(~ind * (y == 1) * gamma_with_h), np.sum(~ind * (y == -1) * gamma_with_h)
        w_l_prev = w_l
        w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
        i += 1
        if i == 10:
            break

    ind = guaranteed_right + (y * w_r < h_l - h_r) * uncertain
    fmargin = y * w_l + h_l + (h_r - h_l + y * w_r) * ind
    loss = np.mean(gamma * np.exp(-fmargin))

    return loss, w_l, w_r


def exp_loss_robust(X_proj, y, gamma, w_l, w_r, w_rs, bs, b_curr, eps, h_flag):
    num = X_proj.shape[0]
    if h_flag:
        h_l, h_r = calc_h(X_proj, y, w_rs, bs, b_curr, eps)
    else:
        h_l, h_r = np.zeros(num), np.zeros(num)

    split_lbs, split_ubs = X_proj - eps, X_proj + eps
    guaranteed_right = split_lbs > b_curr
    uncertain = (split_lbs <= b_curr) * (split_ubs >= b_curr)

    ind = guaranteed_right + (y * w_r < h_l - h_r) * uncertain
    fmargin = y * w_l + h_l + (h_r - h_l + y * w_r) * ind
    loss = np.mean(gamma * np.exp(-fmargin))
    loss = dtype(loss)  # important for the proper selection of the final threshold
    return loss


@jit(nopython=True, nogil=nogil)
def basic_case_two_intervals(y, gamma, guaranteed_right, uncertain, sum_1, sum_m1, max_weight):
    loss_best, w_r_best, w_l_best = np.inf, np.inf, np.inf
    for sign_w_r in (-1, 1):
        # Calculate the indicator function based on the known `sign_w_r`
        ind = guaranteed_right + (y * sign_w_r < 0) * uncertain

        # Calculate all partial sums
        sum_1_1, sum_1_m1 = np.sum(ind * (y == 1) * gamma), np.sum(ind * (y == -1) * gamma)
        sum_0_1, sum_0_m1 = sum_1 - sum_1_1, sum_m1 - sum_1_m1
        # Minimizer of w_l, w_r on the current interval
        w_l, w_r = coord_descent_exp_loss(sum_1_1, sum_1_m1, sum_0_1, sum_0_m1, max_weight)
        # if w_r is on the different side from 0, then sign_w_r*w_r < 0  =>  w_r:=0
        w_r = sign_w_r * max(sign_w_r * w_r, 0)

        # If w_r now become 0, we need to readjust w_l
        if sum_1_m1 != 0 and sum_0_m1 != 0:
            w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
            w_l = clip(w_l, -max_weight, max_weight)
        else:  # to prevent a division over zero
            w_l = max_weight * math.copysign(1, 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1)))

        preds_adv = w_l + w_r * ind

        loss = np.mean(gamma * np.exp(-y * preds_adv))  # also O(n)
        if loss < loss_best:
            loss_best, w_l_best, w_r_best = loss, w_l, w_r
    return loss_best, w_l_best, w_r_best

# def cmp_gap_value(x):
#     return x[0]

def certify_Lp_bound_numba(pre_dp, dp, X, y, threshold_cumu_value, order, precision, C, coords_to_ignore=()):
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
            
            X_proj = X_value[coord]
            if coord in coords_to_ignore:
                continue

            cumu_value = [(a, b, y_sig * value) for (a, b, value) in cumu_value]

            gaps_value = []
            for a, b, value in cumu_value:
                if a >= X_proj:
                    gap = int(((a - X_proj) ** order)/precision)
                elif b < X_proj:
                    gap = int(((X_proj - b) ** order)/precision)
                else:
                    gap = 0
                
                gaps_value.append((gap, value))

            gaps_value.sort()
            
            for j in range(C + 1):
                dp[j] = np.inf
                for gap, value in gaps_value:
                    if j >= gap:
                    # print(value - ori_value)
                        dp[j] = min(dp[j], value + pre_dp[j - gap]) 
                    else:
                        break
            # dp = certify_Lp_bound_iter_feature(pre_dp, dp, X_proj, y_sig, cumu_value, order, precision, C)
            # --------------
            # ---------------
            # pre_dp = np.copy(dp)
            dp, pre_dp = pre_dp, dp
        min_diff[i]=np.copy(pre_dp)
    
    return min_diff


@jit(nopython=True, nogil=nogil)
def certify_Lp_bound_iter_feature(pre_dp, dp, X_proj, y_sig, intervals, values, order, precision, C):
    cumu_value = [(a[0], a[1], y_sig * value) for a, value in zip(intervals, values)]
    # for a, b, value in cumu_value:
    #     if a <= X_proj:
    #         ori_value = value
    #     else:
    #         break

    gaps_value = []
    for a, b, value in cumu_value:
        if a >= X_proj:
            gap = int(((a - X_proj) ** order)/precision)
            # if gap > j:
            #     break
        elif b < X_proj:
            gap = int(((X_proj - b) ** order)/precision)
        else:
            gap = 0
        
        gaps_value.append((gap, value))

    gaps_value.sort()
    
    for j in range(C + 1):
        dp[j] = np.inf
        for gap, value in gaps_value:
            if j >= gap:
            # print(value - ori_value)
                dp[j] = min(dp[j], value + pre_dp[j - gap]) 
            else:
                break
    return dp

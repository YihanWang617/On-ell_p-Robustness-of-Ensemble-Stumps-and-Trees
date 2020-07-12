from verify import verify
from data import breast_cancer
from Lp_xgbKantchelianAttack import main

if __name__ == "__main__":

    X_train, y_train, X_test, y_test, _= breast_cancer()
    robust_bound_LB = verify(X_test, y_test, 'breast_cancer.ensemble.npy', 0.3, 10, 1, precision=0.02)

    args = {'order': 1, 'data': './data/breast_cancer.test', 'model': 'breast_cancer.model', 'num_classes': 2, 'offset': 0, 'num_attacks': 500, 'guard_val': 2e-07, 'round_digits': 20, 'feature_start': 0}
    exact_bound = main(args)

    print(robust_bound_LB <= exact_bound)

    
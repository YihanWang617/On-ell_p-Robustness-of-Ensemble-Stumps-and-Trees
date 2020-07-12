# On L-p Robustness of Decision Stumps and Trees

This repo contains an implementation of **L-p robustness of Ensemble Decision Stumps and Trees**, based on [provably-robust-boosting](https://github.com/max-andr/provably-robust-boosting) and [treeVerification](https://github.com/chenhongge/treeVerification)

## Dependences
### ensemble stump verification and ensemble stump/tree training

- tensorflow
- treelite
- xgboost
- numpy
- gurobi

Please see Lp_stumps_trees/requirements.txt for the detail information.

### ensemble tree verification
- the same as https://github.com/chenhongge/treeVerification

`sudo apt install libuv1-dev libboost-all-dev`

## How to run the code
### Verification

#### L-p stump ensemble verification

```
cd Lp_stumps_trees
```

**verification approximation**

```
python verify.py --dataset breast_cancer --model models/breast_cancer/breast_cancer.ensemble.npy --precision 0.02 --eps 0.3 --num_trials 10 --order 1
```

**xgbKantchelianAttack verification**

(the latest xgboost is not supported)

```
python Lp_xgbKantchelianAttack.py -d=./data/breast_cancer.test -m=./models/breast_cancer/breast_cancer.model -c=2 -p=1 --feature_start 0 --num_attack 500
```

Lp_xgbKantchelianAttack.py only accepts models in xgboost format, and datasets in libsvm format, here we supply a script to transfer models from numpy format to xgboost format and datasets from numpy array to libsvm.

```
python convert_to_libsvm.py --model ./models/breast_cancer/breast_cancer --num_features 10 --feature_start 0 --dataset breast_cancer
```

#### L-p tree ensemble verification

We add a new parameter named *order* based on [treeVerification](https://github.com/chenhongge/treeVerification), which means the order of the norm.

```
cd Lp_treeVerification
./compile.sh
./treeVerify example.json
```

### Training

#### L-p stump ensemble training

```
python train_Lp.py --dataset breast_cancer --weak_learner stump --n_bins 40 --lr 0.4 --n_tree 20 --precision 0.005 --order 1 --schedule_len 4
```

#### L-p tree ensemble training

```
python train_Lp.py --dataset breast_cancer --weak_learner tree --n_bins 40 --lr 1 --n_tree 10  --order 1 --schedule_len 4 --max_depth 3 --sample_size 5000
```
## Models

In `models/` we provide some pretrained models used in our paper.



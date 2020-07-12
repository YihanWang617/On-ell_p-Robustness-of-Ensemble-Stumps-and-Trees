import xgboost as xgb

def convert(model_name, filename):
    bst = xgb.Booster()
    bst.load_model(model_name)

    res = bst.get_dump(dump_format='json')

    with open(filename) as f:
        f.write('[' + ','.join(res) + ']')


convert("../RobustTrees/0010.model", "breast_cancer.json")
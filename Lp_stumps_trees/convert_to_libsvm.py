'''
convert the standard data to libsvm format
'''
import data
import json
import treelite
import xgboost as xgb
import argparse

def convert_data(X, Y, filename):
    res = ""
    for x, y in zip(X, Y):
        res += str(y if y == 1 else 0)
        res += ' '
        for i in range(x.shape[0]):
            res += "{}:{:.8f} ".format(i, x[i])
        res += '\n'

    # print(res)
    with open(filename, 'w+') as f:
        f.write(res)


# X_train, y_train, X_test, y_test, _ = data.breast_cancer()
# convert_data(X_train, y_train, "./data/breast_cancer.train")
# convert_data(X_test, y_test, "./data/breast_cancer.test")

def dfs_build(tree_obj, tree_json, start_feature = 0):
    if "leaf" in tree_json:
        tree_obj[tree_json['nodeid']].set_leaf_node(tree_json["leaf"])
    else:
        # print(tree_json["split"] - start_feature)
        tree_obj[tree_json["nodeid"]].set_numerical_test_node(feature_id=tree_json["split"] - start_feature,
                                                            opname='<',
                                                            threshold=tree_json["split_condition"],
                                                            default_left=True,
                                                            left_child_key=tree_json["children"][0]['nodeid'],
                                                            right_child_key=tree_json["children"][1]['nodeid'])
        dfs_build(tree_obj, tree_json["children"][0], start_feature)
        dfs_build(tree_obj, tree_json["children"][1], start_feature)

def convert_model(model_name, num_feature, start_feature = 0):
    builder = treelite.ModelBuilder(num_feature)
    ensemble = None
    with open(model_name + '.json') as f:
        ensemble = json.load(f)
    for tree_json in ensemble:
        tree = treelite.ModelBuilder.Tree()
        tree[0].set_root()
        dfs_build(tree, tree_json, start_feature)
        builder.append(tree)
    model = builder.commit()

    model.export_as_xgboost(model_name + '.model', 'binary:logistic')

    # model = builder.commit()
    return model
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='breast_cancer', help="name of the model to be transformed.")
    parser.add_argument('--num_features', type=int, default=10)
    parser.add_argument('--feature_start', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='breast_cancer')


    args = parser.parse_args()
    
    convert_model(args.model, args.num_features, args.feature_start)

    X_train, y_train, X_test, y_test, _ = data.all_datasets_dict[args.dataset]()
    convert_data(X_train, y_train, "./data/{}.train".format(args.dataset))
    convert_data(X_test, y_test, "./data/{}.test".format(args.dataset))
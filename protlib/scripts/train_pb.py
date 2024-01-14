import argparse
import os
import sys

import joblib
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-m', '--model-name', type=str)

# parser.add_argument('-t', '--target', type=str)
# parser.add_argument('-hp', '--helpers', type=str)

# parser.add_argument('-tre', '--train-embeds', type=str, nargs='+')
# parser.add_argument('-tse', '--test-embeds', type=str, nargs='+')
#
# parser.add_argument('-b', '--bp', type=int, default=0)
# parser.add_argument('-m', '--mf', type=int, default=0)
# parser.add_argument('-c', '--cc', type=int, default=0)

# parser.add_argument('-cnd', '--conditional', type=str)
# parser.add_argument('-o', '--output', type=str)
parser.add_argument('-d', '--device', type=str)
# parser.add_argument('-g', '--graph', type=str)

if __name__ == '__main__':

    args = parser.parse_args()

    # Optional: set the device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    try:
        from protlib.metric import obo_parser, Graph, get_topk_targets
        from protlib.models.prepocess import get_features_simple, get_targets_from_parquet
        from protlib.models.gbdt import BCEWithNaNLoss, BCEwithNaNMetric

    except ImportError:
        print('Alarm')
        pass

    from py_boost import GradientBoosting
    from py_boost.multioutput.sketching import RandomProjectionSketch

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    model_config = config['base_models'][args.model_name]
    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    embeds_path = os.path.join(config['base_path'], config['embeds_path'])
    helpers_path = os.path.join(config['base_path'], config['helpers_path'])

    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    split = [model_config['bp'], model_config['mf'], model_config['cc']]
    cols = []

    for n, i in enumerate(split):
        cols.extend(get_topk_targets(
            ontologies[n],
            i,
            train_path=os.path.join(config['base_path'], 'Train')
        ))

    fillna = not model_config['conditional']  # args.conditional == 'false'
    print(fillna)
    Y = get_targets_from_parquet(
        os.path.join(helpers_path, 'real_targets'),
        ontologies,
        split,
        ids=cols,
        fillna=fillna
    )

    train_embeds = [os.path.join(embeds_path, x, 'train_embeds.npy') for x in model_config['embeds']]
    test_embeds = [os.path.join(embeds_path, x, 'test_embeds.npy') for x in model_config['embeds']]

    X, train_idx = get_features_simple(
        os.path.join(helpers_path, 'fasta/train_seq.feather'), train_embeds
    )

    X_test, test_idx = get_features_simple(
        os.path.join(helpers_path, 'fasta/test_seq.feather'), test_embeds
    )

    print(X.shape, X_test.shape)

    N_FOLDS = 5
    # assume embedding sum is our key
    key = np.array(list(map(hash, X.sum(axis=1))))
    test_key = np.array(list(map(hash, X_test.sum(axis=1))))

    np.random.seed(42)
    folds = np.unique(key)
    np.random.shuffle(folds)
    folds = np.array_split(folds, N_FOLDS)

    oof_pred = np.zeros((X.shape[0], len(cols)), dtype=np.float32)
    test_pred = np.zeros((X_test.shape[0], len(cols)), dtype=np.float32)

    output = os.path.join(config['base_path'], config['models_path'], args.model_name)
    os.makedirs(output, exist_ok=True)

    for f in range(N_FOLDS):
        # get indexers
        test_sl = np.isin(key, folds[f])

        pred_idx = np.arange(X_test.shape[0])
        tr_idx, ts_idx = np.nonzero(~test_sl)[0], np.nonzero(test_sl)[0]
        print(tr_idx.shape, ts_idx.shape)

        # train model
        model = GradientBoosting(
            BCEWithNaNLoss(), BCEwithNaNMetric(),
            ntrees=20000, lr=0.03, verbose=100, es=300, lambda_l2=10, gd_steps=1,
            subsample=.8, colsample=0.8, min_data_in_leaf=10, use_hess=False,
            max_bin=256, max_depth=6,
            multioutput_sketch=RandomProjectionSketch(3),
            # callbacks=[WarmStart(pretrained)]
        )

        model.fit(X[tr_idx], Y[tr_idx], eval_sets=[{'X': X[ts_idx], 'y': Y[ts_idx]}])
        joblib.dump(model, os.path.join(output, f'model_{f}.pkl'))

        # oof prediction
        oof_pred[ts_idx] += model.predict(X[ts_idx], batch_size=5000)
        # test prediction
        test_pred += model.predict(X_test, batch_size=5000)

    test_pred = test_pred / N_FOLDS

    joblib.dump(oof_pred, os.path.join(output, 'oof_pred.pkl'))
    joblib.dump(test_pred, os.path.join(output, 'test_pred.pkl'))

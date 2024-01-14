import argparse
import os
import sys

import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)

if __name__ == '__main__':

    args = parser.parse_args()

    try:
        from protlib.models.prepocess import get_features_simple

    except ImportError:
        print('Alarm')
        pass

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    embeds_path = os.path.join(config['base_path'], config['embeds_path'])
    helpers_path = os.path.join(config['base_path'], config['helpers_path'])

    train_embeds = [os.path.join(embeds_path, x, 'train_embeds.npy') for x in ['t5']]

    X, train_idx = get_features_simple(
        os.path.join(helpers_path, 'fasta/train_seq.feather'), train_embeds
    )

    print(X.shape, )

    N_FOLDS = 5
    # assume embedding sum is our key
    key = np.array(list(map(hash, X.sum(axis=1))))

    np.random.seed(42)
    folds = np.unique(key)
    np.random.shuffle(folds)
    folds = np.array_split(folds, N_FOLDS)
    
    ff = np.zeros(X.shape[0], dtype=np.int32)

    for n, f in enumerate(folds):
        sl = np.nonzero(np.isin(key, f))[0]
        ff[sl] = n
    np.save(os.path.join(helpers_path, 'folds_gkf.npy'), ff, )

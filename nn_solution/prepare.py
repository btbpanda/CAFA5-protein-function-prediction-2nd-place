import argparse
import time
from scipy import sparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import scipy
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)

# DATA_DIR1 = sys.argv[1]
# DATA_DIR2 = sys.argv[2]
# DATA_DIR3 = sys.argv[3]

#DATA_DIR1 = "/kaggle/input/cafa-5-protein-function-prediction/"
#DATA_DIR2 = "/kaggle/input/t5embeds/"
#DATA_DIR3 = './'

n_labels_to_consider = 31466 # We will choose only top frequent labels (in train) and predict only them. 

if __name__ == '__main__':
    args = parser.parse_args()
    
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    
    base_path = config['base_path']
    feats_path = os.path.join(base_path, config['helpers_path'], 'feats')
    os.makedirs(feats_path, exist_ok=True)

    trainTerms = pd.read_csv(os.path.join(base_path, 'Train', 'train_terms.tsv'),sep="\t")
    print(trainTerms.shape)
    vec_freqCount = (trainTerms['term'].value_counts())
    print(vec_freqCount)

    labels_to_consider = np.array(list(vec_freqCount.index[:n_labels_to_consider] ))
    # np.save(os.path.join(DATA_DIR3,'Y_31466_labels.npy'), labels_to_consider)
    np.save(os.path.join(feats_path,'Y_31466_labels.npy'), labels_to_consider)

    # xxs = np.load(os.path.join(DATA_DIR2, 'train_ids.npy'))
    xxs = pd.read_feather(
        os.path.join(config['base_path'], config['helpers_path'], 'fasta/train_seq.feather'),
        columns=['EntryID']
    )['EntryID'].values

    trainTerms['x'] = trainTerms['EntryID'].map({x:i for i,x in enumerate(xxs)})
    trainTerms['y'] = trainTerms['term'].map({x:i for i,x in enumerate(labels_to_consider)})

    mat = scipy.sparse.coo_matrix((np.ones(len(trainTerms)), 
                                   (trainTerms['x'].values, trainTerms['y'].values)
                                  )
                                 ).tocsr().astype(np.float32)
    # scipy.sparse.save_npz(os.path.join(DATA_DIR3,'Y_31466_sparse_float32.npz'), mat)
    scipy.sparse.save_npz(os.path.join(feats_path,'Y_31466_sparse_float32.npz'), mat)

    # train_terms = pd.read_csv(os.path.join(DATA_DIR1, 'Train', 'train_terms.tsv'), sep='\t')
    train_terms = pd.read_csv(os.path.join(base_path, 'Train', 'train_terms.tsv'), sep='\t')
    terms = train_terms.groupby(['aspect', 'term'])['term'].count().reset_index(name='frequency')
    print(terms.groupby('aspect')['term'].nunique())

    CCOProt = set(train_terms[train_terms['aspect']=='CCO']['EntryID'].unique())
    MFOProt = set(train_terms[train_terms['aspect']=='MFO']['EntryID'].unique())
    BPOProt = set(train_terms[train_terms['aspect']=='BPO']['EntryID'].unique())

    AllProt = set(train_terms['EntryID'].unique())
    FullProt = []
    for x in MFOProt:
        if x in CCOProt and x in BPOProt:
            FullProt.append(x)
    len(FullProt)
    FullProt = set(FullProt)

    r = []
    for t in xxs:
        if t in FullProt:
            r.append(t)
    # np.save(os.path.join(DATA_DIR3,'train_ids_cut43k.npy'), np.array(r))
    np.save(os.path.join(feats_path,'train_ids_cut43k.npy'), np.array(r))
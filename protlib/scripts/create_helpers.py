import argparse
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tqdm
import yaml
from numba import njit, prange
from pandas import Series, DataFrame

print(os.path.abspath(os.path.join(__file__, '../../../')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

try:
    from protlib.metric import obo_parser, Graph, ia_parser, get_funcs_mapper
except Exception:
    obo_parser, Graph, ia_parser, get_funcs_mapper = None, None, None, None

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-b', '--batch-size', type=int)
parser.add_argument('-p', '--propagate', type=bool, default=False)


@njit
def prop_max_cpu(mat, k, adj):
    for i in prange(mat.shape[0]):
        if mat[i, k] == 1:
            continue

        for j in adj:
            if mat[i, j] == 1:
                mat[i, k] = 1
                continue

    return


def propagate_target(mat, G):
    for f in G.order:

        adj = G.terms_list[f]['children']

        if len(adj) == 0:
            continue

        prop_max_cpu(mat, f, np.asarray(adj))

    return


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    path = os.path.join(config['base_path'], config['helpers_path'], 'real_targets')
    os.makedirs(path, exist_ok=True)

    trainTerms = pd.read_csv(os.path.join(config['base_path'], 'Train/train_terms.tsv'), sep='\t')

    terms = trainTerms.set_index('EntryID')
    terms['namespace'] = terms['aspect'].map(
        {'BPO': 'biological_process', 'MFO': 'molecular_function', 'CCO': 'cellular_component'}
    )

    vec_train_protein_ids = pd.read_feather(
        os.path.join(config['base_path'], config['helpers_path'], 'fasta/train_seq.feather'),
        columns=['EntryID'],
    )['EntryID'].values

    ia_dict = ia_parser(os.path.join(config['base_path'], 'IA.txt'))
    ontologies = []
    for ns, terms_dict in obo_parser(os.path.join(config['base_path'], 'Train/go-basic.obo')).items():
        ontologies.append(Graph(ns, terms_dict, ia_dict, True))

    for n, i in tqdm.tqdm(enumerate(range(0, vec_train_protein_ids.shape[0], args.batch_size))):

        idx = vec_train_protein_ids[i: i + args.batch_size]
        num = Series(np.arange(idx.shape[0]), index=idx)
        trm = terms.loc[idx]

        # reformat targets
        for ont in ontologies:

            os.makedirs(os.path.join(path, ont.namespace), exist_ok=True)

            trm_ont = trm.query(f"namespace == '{ont.namespace}'").copy()
            trm_ont['id'] = trm_ont['term'].map(get_funcs_mapper(ont)).values
            trm_ont['n'] = num.loc[trm_ont.index].values

            trg = np.zeros((num.shape[0], ont.idxs), dtype=np.float32)
            np.add.at(trg, (trm_ont['n'].values, trm_ont['id'].values), 1)

            if args.propagate:
                propagate_target(trg, ont)

            # create NaNs from graph
            for k, node in enumerate(ont.terms_list):
                adj = node['adj']
                if len(adj) > 0:
                    na = np.nonzero(np.nansum(trg[:, adj], axis=1) == 0)[0]
                    assert np.nansum(trg[na, k]) == 0, 'Should be empty'
                    trg[na, k] = np.nan

            trg = DataFrame(trg, columns=[x['id'] for x in ont.terms_list])
            trg['EntryID'] = idx
            trg.to_parquet(os.path.join(path, ont.namespace, f'part_{str(n).zfill(2)}.parquet'))
    
    # count priors
    for ont in ontologies:
        trg = pd.read_parquet(glob.glob(os.path.join(path, ont.namespace, f'part_*')),
                              columns=[x['id'] for x in ont.terms_list])
        mean = trg.mean().fillna(0).values
        nulls = trg.isnull().mean().values

        joblib.dump(mean, os.path.join(path, ont.namespace, f'prior.pkl'))
        joblib.dump(nulls, os.path.join(path, ont.namespace, f'nulls.pkl'))

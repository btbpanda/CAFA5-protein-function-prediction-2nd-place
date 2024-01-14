import argparse
import os

import pandas as pd
import tqdm
import yaml

# from.create_helpers import

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str)
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-o', '--output', type=str, default='./')

# parser.add_argument('-p', '--path', type=str)
# parser.add_argument('-f', '--fasta', type=str)
# parser.add_argument('-o', '--output', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    train_idx = set(
        pd.read_feather(
            os.path.join(config['base_path'], 'helpers/fasta/train_seq.feather'),
            columns=['EntryID']
        )['EntryID']
    )
    test_idx = set(
        pd.read_feather(
            os.path.join(config['base_path'], 'helpers/fasta/test_seq.feather'),
            columns=['EntryID']
        )['EntryID']
    )
    idxs = train_idx.union(test_idx)

    reader = pd.read_csv(
        os.path.join(config['base_path'], config['temporal_path'], args.file),
        sep='\t',
        header=None,
        names=['x', 'EntryID', 'xx', 'type', 'term', 'y', 'source', 'yyy', 'z', 'zz', 'zzz', 'a', 'aa', 'date', 'b',
               'bb', 'bbb'],
        usecols=['EntryID', 'term', 'source', ],
        chunksize=1_000_000,
        na_filter=True
    )

    store = []

    for n, batch in tqdm.tqdm(enumerate(reader)):

        if n == 0:
            batch = batch.dropna()

        filtred = batch[(batch['EntryID'].isin(idxs))]
        filtred = filtred[['EntryID', 'term', 'source',]].drop_duplicates()

        if len(store) > 0 and len(filtred) > 0 and \
                (store[-1].iloc[-1].values == filtred.iloc[0].values).all():
            filtred = filtred[1:]

        store.append(filtred)

    store = pd.concat(store, ignore_index=True)

    output = os.path.join(config['base_path'], config['temporal_path'], args.output)

    os.makedirs(output, exist_ok=True)
    store.to_csv(os.path.join(output, 'goa.tsv', ), index=False, sep='\t')

    # split by evidence
    data = store.drop_duplicates()
    path = os.path.join(output, 'labels')
    os.makedirs(path, exist_ok=True)

    kaggle_codes = set('IDA IMP TAS IPI IEP IGI IC EXP HTP HDA HMP HGI HEP'.split())

    data[(data['EntryID']).isin(test_idx - train_idx) & (data['source'].isin(kaggle_codes))].to_csv(
        f'{path}/test_leak_no_dup.tsv', sep='\t', index=False)
    data[(data['EntryID']).isin(train_idx) & (~data['source'].isin(kaggle_codes))].to_csv(
        f'{path}/train_no_kaggle.tsv', sep='\t', index=False)
    data[(data['EntryID']).isin(test_idx) & (~data['source'].isin(kaggle_codes))].to_csv(
        f'{path}/test_no_kaggle.tsv', sep='\t', index=False)

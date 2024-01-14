import argparse
import os

import pandas as pd
import tqdm
import yaml
from pandas import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)


def create_train(path, output):
    df = []
    d_row, seq = None, None
    tax = pd.read_csv(os.path.join(path, 'train_taxonomy.tsv'), sep='\t', index_col='EntryID').squeeze()

    with open(os.path.join(path, 'train_sequences.fasta')) as f:

        for n, row in tqdm.tqdm(enumerate(f.readlines())):

            # break
            if row.startswith('>'):

                # save if not first
                if d_row is not None:
                    d_row['seq'] = seq
                    df.append(d_row)

                # initialize d_row and empty seq
                d_row, seq = {}, ''
                # parse first line
                d_row['EntryID'], row = row[1:].split(' ', 1)
                d_row['taxonomyID'] = tax[d_row['EntryID']]
                d_row['source'], _, row = row.split('|', 2)
                d_row['gene_name'], row = row.split(' ', 1)

                k = 'descr'

                for knext in ['OS', 'OX', 'GN', 'PE', 'SV']:
                    sp = row.split(knext + '=', 1)
                    if len(sp) == 2:
                        v, row = sp
                        d_row[k] = v[:-1]
                        if k in ['OX', 'PE', 'SV']:
                            d_row[k] = float(d_row[k])
                        k = knext

                d_row[k] = row[:-1]

            else:
                seq += row[:-1]

    # save last
    d_row['seq'] = seq
    df.append(d_row)
    df = DataFrame(df)

    df.to_feather(os.path.join(output, 'train_seq.feather'))

    return


def create_test(path, output):
    df = []
    d_row, seq = None, None

    with open(os.path.join(path, 'testsuperset.fasta')) as f:

        for n, row in tqdm.tqdm(enumerate(f.readlines())):

            if row.startswith('>'):

                # save if not first
                if d_row is not None:
                    d_row['seq'] = seq
                    df.append(d_row)

                # initialize d_row and empty seq
                d_row, seq = {}, ''
                # parse first line
                d_row['EntryID'], d_row['taxonomyID'] = row[1:-1].split('\t', 1)
                d_row['taxonomyID'] = int(d_row['taxonomyID'])

            else:
                seq += row[:-1]

    # save last
    d_row['seq'] = seq
    df.append(d_row)
    df = DataFrame(df)

    df.to_feather(os.path.join(output, 'test_seq.feather'))

    return


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    helpers_path = os.path.join(config['base_path'], config['helpers_path'])
    train_path = os.path.join(config['base_path'], 'Train/')
    test_path = os.path.join(config['base_path'], 'Test (Targets)/')
    output_path = os.path.join(helpers_path, 'fasta')

    os.makedirs(output_path, exist_ok=True)
    create_train(train_path, output_path)
    create_test(test_path, output_path)

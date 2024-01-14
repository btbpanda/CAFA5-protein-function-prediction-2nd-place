import argparse
import os
import sys

import pandas as pd

print(os.path.abspath(os.path.join(__file__, '../../../')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-g', '--graph', type=str)

if __name__ == '__main__':

    try:
        from protlib.metric import obo_parser, Graph
    except Exception:
        obo_parser, Graph = [None] * 2

    args = parser.parse_args()
    old = pd.read_csv(os.path.join(args.path, 'old214', 'goa.tsv'), sep='\t', usecols=['EntryID', 'term']) \
        .drop_duplicates()

    # get quickgo51 dataset
    idxs = {
        'O13397', 'O13398', 'Q7XB51', 'C1L360', 'O44074', 'A0A1D6E0S8', 'A0A0J9UVG7', 'A0A144A2H0', 'Q33862',
        'A5K3U9', 'A0A0D2Y5A7', 'C4M4T9', 'K1XVG1', 'Q8I2J3', 'G2X7W6', 'K1WG73', 'K1XT82', 'K1XW16', 'C0HM68',
        'J9VQH1', 'I1RAQ3', 'A0A509AGE2', 'Q8IJH8', 'Q2T6X7', 'A0A1S4F020', 'I1REI8', 'A0A8V1ABE9', 'A0A125YZN2',
        'C0HLS4', 'A0A7M6UUR2', 'A0A125YHX7', 'O08546', 'C4M633', 'I1RET0', 'Q503K9', 'A0A7M7H308', 'A0A0A7LRQ7'
    }

    qg51 = old[old['EntryID'].isin(idxs)]
    qg51.to_csv(os.path.join(args.path, 'quickgo51.tsv'), sep='\t', index=False)

    # get cafa-terms-diff
    new = pd.read_csv(os.path.join(args.path, 'goa.tsv'), sep='\t', usecols=['EntryID', 'term']) \
        .drop_duplicates()
    new['flg'] = 1

    merged = pd.merge(old, new, on=['EntryID', 'term'], how='left')
    merged = merged[merged['flg'].isnull()].drop('flg', axis=1)

    # get valid term idxs
    valid_idxs = set()
    for ns, terms_dict in obo_parser(args.graph).items():
        G = Graph(ns, terms_dict, None, True)
        valid_idxs = valid_idxs.union([x['id'] for x in G.terms_list])

    merged = merged[merged['term'].isin(valid_idxs)]
    merged['prob'] = 0.99
    merged.to_csv(os.path.join(args.path, 'cafa-terms-diff.tsv'), sep='\t', index=False, header=False)

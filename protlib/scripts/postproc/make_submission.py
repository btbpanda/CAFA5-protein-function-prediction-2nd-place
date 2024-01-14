import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str, default="1")
parser.add_argument('-mr', '--max-rate', type=float, default=0.5)

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import cudf

    try:
        from protlib.metric import CAFAMetric
        from protlib.cafa_utils import obo_parser, Graph
    except Exception:
        CAFAMetric, obo_parser, Graph = [None] * 3

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    temporal_path = os.path.join(config['base_path'], config['temporal_path'])
    ia_path = os.path.join(config['base_path'], 'IA.txt')
    pp_path = os.path.join(config['base_path'], config['models_path'], 'postproc')
    pp_min = os.path.join(pp_path, f'pred_min.tsv')
    pp_max = os.path.join(pp_path, f'pred_max.tsv')
    sub_path = os.path.join(config['base_path'], 'sub')
    os.makedirs(sub_path, exist_ok=True)

    # aggregate min prop and max prop
    pred_max = cudf.read_csv(pp_max, sep='\t', header=None, names=['EntryID', 'term', 'prob'])
    pred_min = cudf.read_csv(pp_min, sep='\t', header=None, names=['EntryID', 'term', 'prob'])
    pred = cudf.merge(pred_max, pred_min, on=['EntryID', 'term'], how='outer').fillna(0)
    pred['prob'] = pred['prob_x'] * args.max_rate + pred['prob_y'] * (1 - args.max_rate)
    pred = pred[['EntryID', 'term', 'prob']]

    label_path = os.path.join(config['base_path'], config['models_path'], 'gcn/bp/temp/labels.tsv')
    if os.path.exists(label_path):
        pred.to_csv(
            os.path.join(sub_path, 'submission.tsv'), header=False, index=False, sep='\t'
        )
        metric = CAFAMetric(
            graph_path,
            ia_path,
            None,
            prop_mode='fill',
            batch_size=30000
        )

        score = metric.from_df(label_path, os.path.join(sub_path, 'submission.tsv'))
        print('CAFA5 Scores')
        print(score)

    # make submission part
    mapper = []
    for n, (ns, terms_dict) in enumerate(obo_parser(graph_path).items()):
        G = Graph(ns, terms_dict, None, True)
        terms = [x['id'] for x in G.terms_list]
        mapper.append(pd.Series([n] * len(terms), index=terms))

    mapper = cudf.from_pandas(pd.concat(mapper))

    # goa leak
    goa = cudf.read_csv(
        os.path.join(temporal_path, 'labels/prop_test_leak_no_dup.tsv'),
        sep='\t', usecols=['EntryID', 'term'])
    goa['prob'] = 0.99

    # gq51 dataset
    qg = cudf.read_csv(
        os.path.join(temporal_path, 'prop_quickgo51.tsv'),
        sep='\t', usecols=['EntryID', 'term'])
    qg['prob'] = 0.99

    # diff
    diff = cudf.read_csv(
        os.path.join(temporal_path, 'cafa-terms-diff.tsv'),
        header=None, sep='\t', names=['EntryID', 'term', 'prob']
    )

    # collect all together
    pred = cudf.concat([pred, qg, goa, diff], ignore_index=False)
    pred['ns'] = pred['term'].map(mapper).values
    pred = pred.groupby(['EntryID', 'term']).mean().reset_index()
    pred['rank'] = pred.groupby(['EntryID', 'ns'])['prob'].rank(method='dense', ascending=False) - 1
    pred = pred.query('rank < 500')

    pred[['EntryID', 'term', 'prob']].to_csv(
        os.path.join(sub_path, 'submission.tsv'), header=False, index=False, sep='\t'
    )

import argparse
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))
print(sys.executable)
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--ontology', type=str)
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str)

ont_dict = {

    'bp': 0,
    'mf': 1,
    'cc': 2

}

if __name__ == '__main__':

    args = parser.parse_args()
    # Optional: set the device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    import torch

    try:
        from protnn.utils import get_labels, get_goa_data, CAFAEvaluator
        from protnn.dataset import StackDataset, StackDataLoader
        from protnn.stacker import GCNStacker
        from protnn.swa import SWA
        from protnn.train import train

        from protlib.metric import obo_parser, Graph, ia_parser
        from protlib.metric import get_topk_targets

    except ImportError:
        print('Alarm')
        pass

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    NOUT = ont_dict[args.ontology]  # ontology to train
    # create paths
    models_path = os.path.join(config['base_path'], config['models_path'])
    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    ia_path = os.path.join(config['base_path'], 'IA.txt')
    helpers_path = os.path.join(config['base_path'], config['helpers_path'])
    temporal_path = os.path.join(config['base_path'], config['temporal_path'])

    work_dir = os.path.join(models_path, 'gcn', args.ontology)
    temp_dir = os.path.join(work_dir, 'temp')
    swa_dir = os.path.join(work_dir, 'swa')
    os.makedirs(temp_dir, exist_ok=True)  # temp path to store some data

    # get graph
    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, ia_parser(ia_path), True))

    G = ontologies[NOUT]
    root_id = [x['id'] for x in G.terms_list if len(x['adj']) == 0][0]

    # get train data
    flist = sorted(glob.glob(os.path.join(helpers_path, f'real_targets/{G.namespace}/part*')))
    target = pd.concat([
        pd.read_parquet(x, columns=[x['id'] for x in G.terms_list]) for x in flist
    ], ignore_index=True).fillna(0)  # pd.read_parquet(flist, columns=names_)

    train_sl = np.nonzero(target[root_id].values == 1)[0]
    target = target.values[train_sl]

    # get priors
    prior_cnd = joblib.load(os.path.join(helpers_path, f'real_targets/{G.namespace}/prior.pkl'))
    nulls = joblib.load(os.path.join(helpers_path, f'real_targets/{G.namespace}/nulls.pkl'))
    prior_raw = prior_cnd * (1 - nulls)

    # get ids
    prot_id = pd.read_feather(
        os.path.join(helpers_path, 'fasta/train_seq.feather'),
        columns=['EntryID']
    )['EntryID'].values

    prot_id = prot_id[train_sl]
    # get goa annotation features
    goa_data = get_goa_data(temporal_path, 'train', prot_id, G)

    # get model config

    # models_config = [
    #     # model path           # split by out     # if was trained with NaN (conditional prob)
    #     ['../pb_t54500_cond/', [3000, 1000, 500], True],
    #     ['../pb_t54500_raw/', [3000, 1000, 500], False],
    #     ['../lin_t5_cond/', [10000, 2000, 1500], True],
    #     ['../lin_t5_raw/', [10000, 2000, 1500], False],
    # ]
    nn_cfg = config['gcn'][args.ontology]
    models_config = []

    for mod in nn_cfg['preds']:
        models_config.append([
            os.path.join(models_path, mod),
            [
                config['base_models'][mod]['bp'],
                config['base_models'][mod]['mf'],
                config['base_models'][mod]['cc']
            ],
            config['base_models'][mod]['conditional']
        ])

    # get train features from base models
    preds = []
    qs = []

    for folder, split, cnd in models_config:
        path = os.path.join(folder, 'oof_pred.pkl')
        oof_pred = joblib.load(path)[train_sl][:, sum(split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]

        idx = get_topk_targets(G, split[NOUT], train_path=os.path.join(config['base_path'], 'Train/'))
        preds.append((oof_pred, idx, cnd))

    # create validation set as the leakage part of test set

    test_labels = pd.concat([
        pd.read_csv(os.path.join(temporal_path, 'labels/prop_test_leak_no_dup.tsv'), sep='\t'),
        pd.read_csv(os.path.join(temporal_path, 'prop_quickgo51.tsv'), sep='\t'),
    ]).drop_duplicates().reset_index(drop=True)

    # save targets for evaluator
    test_labels.to_csv(os.path.join(temp_dir, 'labels.tsv'), index=False, sep='\t')

    ids_to_take = test_labels['EntryID'].drop_duplicates().values

    # targets
    sl = pd.read_feather(os.path.join(helpers_path, 'fasta/test_seq.feather'), columns=['EntryID']) \
        .reset_index() \
        .set_index('EntryID') \
        .loc[ids_to_take]

    ids_to_take = sl.index.values
    sl = sl.values[:, 0]

    # goa data

    test_goa_data = get_goa_data(temporal_path, 'test', ids_to_take, G)
    # features
    test_preds = []

    for n, (folder, split, cnd) in enumerate(models_config):
        path = os.path.join(folder, 'test_pred.pkl')
        test_pred = joblib.load(path)[sl][:, sum(split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]

        idx = get_topk_targets(G, split[NOUT], train_path=os.path.join(config['base_path'], 'Train/'))
        test_preds.append((test_pred, idx, cnd))

    # add side models prediction
    for side_pred in nn_cfg['side_preds']:
        side_pred = os.path.join(
            models_path,
            side_pred,
            config['public_models'][side_pred]['source'],
        )

        side_pred = joblib.load(side_pred)
        split = side_pred['borders']
        oof_pred = side_pred['pred'][train_sl][:, sum(split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]

        test_pred = side_pred['test_pred'][sl][:, sum(split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]

        idx = side_pred['idx'][sum(split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]
        preds.append((oof_pred, idx, False))
        test_preds.append((test_pred, idx, False))

    # evaluator - used to calc metric in other python env
    evaluator = CAFAEvaluator(
        os.path.join(config['base_path'], config['rapids-env']),
        os.path.join(temp_dir, 'labels.tsv'),  # targets path
        config['base_path'],  # path to import protlib
        os.path.join(config['base_path'], 'Train/go-basic.obo'),
        os.path.join(config['base_path'], 'IA.txt'),
        ids_to_take,  # ids to use
        G,  # parsed graph
        batch_size=3000,
        device=args.device,
        temp_dir=temp_dir  # temp dir to store csvs
    )
    # define datasets and data loaders
    train_ds = StackDataset(
        preds,
        G.idxs,
        prior_raw,
        prior_cnd,
        G,
        goa_list=goa_data,
        p_goa=.5,
        targets=target
    )
    train_dl = StackDataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)

    val_ds = StackDataset(
        test_preds,
        G.idxs,
        prior_raw,
        prior_cnd,
        G,
        goa_list=test_goa_data,
        p_goa=1,
        targets=None
    )
    val_dl = StackDataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8)

    # define the model
    model = GCNStacker(
        5, 1,
        G.idxs,
        hidden_size=nn_cfg['hidden_size'],
        n_layers=nn_cfg['n_layers'],
        embed_size=nn_cfg['embed_size']
    ).cuda()

    swa = SWA(nn_cfg['store_swa'], path=swa_dir, rewrite=True)  # 10 best checkpoints are saved

    model, swa, scores = train(
        model,
        swa,
        train_dl,
        val_dl,
        evaluator,
        n_ep=nn_cfg['n_ep'], lr=1e-3, clip_grad=1e-1, weight_decay=0  # 3 epochs for example, I use 20
    )
    joblib.dump(swa, os.path.join(work_dir, f'swa.pkl'))

    # validate SWA
    model = swa.set_weights(model, 3, weighted=False)
    torch.save(model.state_dict(), os.path.join(work_dir, f'checkpoint.pth'))
    
    score = evaluator(model, val_dl)
    print('Final CAFA5 score', score)

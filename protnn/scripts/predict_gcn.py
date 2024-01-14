import argparse
import os
import sys

import joblib
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str)

if __name__ == '__main__':

    args = parser.parse_args()
    # Optional: set the device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import torch

    try:
        from protnn.utils import get_labels, get_goa_data, CAFAEvaluator, make_submission
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

    ontologies = []
    for ns, terms_dict in obo_parser(os.path.join(config['base_path'], 'Train/go-basic.obo')).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    test_idx = pd.read_feather(
        os.path.join(config['base_path'], 'helpers/fasta/test_seq.feather'),
        columns=['EntryID']
    )['EntryID'].values

    # make prediction
    for nout, ontology in enumerate(['bp', 'mf', 'cc']):
        mode = 'w' if nout == 0 else 'a'
        nn_cfg = config['gcn'][ontology]
        work_dir = os.path.join(config['base_path'], 'models/gcn', ontology)
        models_path = os.path.join(config['base_path'], config['models_path'])

        G = ontologies[nout]

        # load models
        model = GCNStacker(
            5, 1,
            G.idxs,
            hidden_size=nn_cfg['hidden_size'],
            n_layers=nn_cfg['n_layers'],
            embed_size=nn_cfg['embed_size']
        )
        model.load_state_dict(torch.load(os.path.join(work_dir, 'checkpoint.pth')))
        model = model.cuda()

        # shuffle some model ids
        for k, tta_cfg in enumerate(nn_cfg['tta']):
            output_path = os.path.join(config['base_path'], 'models/gcn', f'pred_tta_{k}.tsv')
            model_ids = nn_cfg['tta'][tta_cfg]

            models_config = []

            for mod in model_ids:
                models_config.append([
                    os.path.join(models_path, mod),
                    [
                        config['base_models'][mod]['bp'],
                        config['base_models'][mod]['mf'],
                        config['base_models'][mod]['cc']
                    ],
                    config['base_models'][mod]['conditional']
                ])
                
            print(models_config)

            # get features
            test_preds = []


            prior_cnd = joblib.load(
                os.path.join(config['base_path'], f'helpers/real_targets/{G.namespace}/prior.pkl')
            )
            nulls = joblib.load(
                os.path.join(config['base_path'], f'helpers/real_targets/{G.namespace}/nulls.pkl')
            )
            prior_raw = prior_cnd * (1 - nulls)

            for folder, split, cnd in models_config:
                path = os.path.join(folder, 'test_pred.pkl')
                test_pred = joblib.load(path)[:, sum(split[:nout]): sum(split[:nout]) + split[nout]]
                idx = get_topk_targets(G, split[nout], train_path=os.path.join(config['base_path'], 'Train/'))
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

                test_pred = side_pred['test_pred'][:, sum(split[:nout]): sum(split[:nout]) + split[nout]]

                idx = side_pred['idx'][sum(split[:nout]): sum(split[:nout]) + split[nout]]
                test_preds.append((test_pred, idx, False))

            test_goa_data = get_goa_data(os.path.join(config['base_path'], 'temporal'), 'test', test_idx, G)
            
            test_ds = StackDataset(
                test_preds,
                G.idxs,
                prior_raw,
                prior_cnd,
                G,
                goa_list=test_goa_data,
                p_goa=1,
                targets=None
            )
            test_dl = StackDataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8)

            make_submission(
                model,
                test_dl,
                G,
                test_idx,
                output_path,
                mode=mode,
                topk=500,
                tau=0.01
            )

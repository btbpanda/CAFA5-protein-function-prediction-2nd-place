import os
import subprocess

import numpy as np
import pandas as pd
import torch
import tqdm

try:
    from protlib.metric import get_funcs_mapper, get_topk_targets, get_depths, get_ns_id
    from protlib.cafa_utils import Graph, obo_parser
except ImportError:
    pass


def get_labels(path, G, idx, src=None):
    ns = get_ns_id(G)[1].upper() + 'O'
    data = pd.read_csv(
        path, sep='\t'
    )
    # .query(f'aspect == "{ns}"')
    data = data[data['term'].isin(set([x['id'] for x in G.terms_list]))].copy()

    if src is not None:
        data = data.query(f'source == "{src}"').copy()

    data['term'] = data['term'].map(get_funcs_mapper(G)).values
    data = data.groupby('EntryID')['term'].agg(list)

    diff = list(set(idx) - set(data.index))
    diff = pd.Series([[]] * len(diff), index=diff, name='term')

    data = pd.concat([data, diff]).loc[idx]

    return data


def get_goa_data(path, pref, ids, G):
    goa_data = []
    # src_list = ['IEA', 'IBA', 'ISO', 'ISS', 'NAS', 'ND', 'ISM', 'RCA', 'ISA', 'IKR', 'IGC']
    src_list = ['no_kaggle']

    for src in src_list:
        fname = os.path.join(path, f'labels/prop_{pref}_{src}.tsv')
        goa_data.append(
            get_labels(
                fname,
                G,
                ids,
            )
        )

    return goa_data


def make_raw_prediction(model, dl):
    model.eval()

    out_shape = (dl.dataset.preds[0][0].shape[0], dl.dataset.nout)
    pred = np.zeros(out_shape, dtype=np.float32)

    start = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            pred[start: start + dl.batch_size] = model(batch).sigmoid().detach().cpu().numpy()
            start += dl.batch_size

    return pred


def make_submission(model, dl, G, idx, path, mode='w', topk=500, tau=0.01):
    model.eval()
    terms = np.array([x['id'] for x in G.terms_list])
    idx = np.array(idx)
    flg = mode == 'w'

    with torch.no_grad():
        for n, batch in enumerate(tqdm.tqdm(dl)):
            pred = model(batch).sigmoid()

            order = pred.argsort(dim=1, descending=True)[:, :topk]
            pred = torch.gather(pred, 1, order).detach().cpu().numpy().ravel()
            order = order.detach().cpu().numpy().ravel()

            sub = pd.DataFrame({

                'EntryID': idx[n * dl.batch_size: (n + 1) * dl.batch_size].repeat(topk),
                'term': terms[order],
                'prob': pred

            })

            sub = sub.query(f'prob >= {tau}').copy()
            sub['prob'] = sub['prob'].astype('str').str.slice(0, 5)

            with open(path, 'w' if flg else 'a') as f:
                sub.to_csv(f, header=False, index=False, sep='\t')
                flg = False

    return


script = """
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', type=str)
parser.add_argument('-f', '--pred', type=str)
parser.add_argument('-g', '--graph', type=str)
parser.add_argument('-i', '--ia', type=str)
parser.add_argument('-p', '--protlib', type=str)
parser.add_argument('-b', '--batch-size', type=int)
parser.add_argument('-o', '--output', type=str)
parser.add_argument('-d', '--device', type=int)

if __name__ == '__main__':

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    sys.path.append(args.protlib)
    from protlib.metric import CAFAMetric, get_ns_id

    metric = CAFAMetric(
        args.graph,
        args.ia,
        None,
        prop_mode='fill',
        batch_size=args.batch_size
    )

    scores = metric.from_df(args.target, args.pred)
    score = scores[list(scores.keys())[0]]


    with open(args.output, 'w') as f:
        f.write(str(score))

"""


class CAFAEvaluator:

    def __init__(
            self,
            rapids_env,
            target_path,
            protlib_path,
            graph_path,
            ia_path,
            idx,
            G,
            device=0,
            temp_dir='temp_dir',
            batch_size=3000,
            script=script
    ):
        self.rapids_env = rapids_env
        self.temp_dir = temp_dir
        self.protlib_path = protlib_path
        self.graph_path = graph_path
        self.ia_path = ia_path
        self.batch_size = batch_size
        self.G = G
        self.idx = idx
        self.device = device

        targets = pd.read_csv(target_path, sep='\t')
        targets = targets[
            (targets['EntryID'].isin(set(idx))) &
            (targets['term'].isin(set([x['id'] for x in G.terms_list])))
            ]

        os.makedirs(temp_dir, exist_ok=True)
        self.target_path = os.path.join(temp_dir, 'train_terms.tsv')
        targets.to_csv(self.target_path, index=False, sep='\t')

        with open(os.path.join(temp_dir, 'evaluator.py'), 'w') as f:
            f.write(script)

    def __call__(self, model, dl, topk=500, tau=0.01):
        sub_file = os.path.join(self.temp_dir, 'sub.tsv')
        make_submission(
            model, dl, self.G, self.idx, sub_file, mode='w', topk=topk, tau=tau
        )

        tmp_file = os.path.join(self.temp_dir, 'temp_score.txt')
        target_file = os.path.join(self.temp_dir, 'train_terms.tsv')

        call = f"""{self.rapids_env} {self.temp_dir}/evaluator.py \
            --target {target_file} \
            --pred  {sub_file} \
            --graph {self.graph_path} \
            --ia {self.ia_path} \
            --protlib {self.protlib_path} \
            --batch-size {self.batch_size} \
            --output {tmp_file} \
            --device {self.device}"""

        subprocess.check_call(call, shell=True)

        with open(tmp_file) as f:
            score = float(f.readline())

        return score

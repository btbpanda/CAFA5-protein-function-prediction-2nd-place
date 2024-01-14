import glob
import os

import numpy as np
import pandas as pd

from ..metric import get_funcs_mapper


def get_tax(fasta):
    tax_list = [
        9606, 3702, 10090, 7955, 7227, 10116, 559292, 6239,
        284812, 83333, 83332, 44689, 237561, 39947, 9031, 36329,
        9913, 227321, 8355, 9823, 224308, 330879, 4577, 170187,
        9615, 99287, 85962, 243232, 287, 235443, 8364
    ]

    tax = fasta['taxonomyID'].map(
        {x: n for (n, x) in enumerate(tax_list, 1)}
    ).fillna(0).astype(np.int32).values[:, np.newaxis]
    tax = tax == np.arange(len(tax_list) + 1)[np.newaxis, :]
    tax = tax.astype(np.float32)

    return tax


def get_sergey_embeds(fasta, path, ):
    embed = np.load(path).astype(np.float32)

    dirname, basename = os.path.dirname(path), os.path.basename(path)
    id_name = os.path.join(dirname, basename.replace('_embeds', '_ids'))

    idx = np.load(id_name)

    if (len(idx) == len(fasta)) and (np.asarray(idx) == fasta['EntryID'].values).all():
        return embed

    idx = pd.Series(np.arange(idx.shape[0]), index=idx)
    idx = idx[fasta['EntryID'].values]

    return embed[idx]


def get_features_simple(fasta, embeds_list):
    fasta = pd.read_feather(fasta)
    tax = get_tax(fasta)
    embeds = np.concatenate([get_sergey_embeds(fasta, x) for x in embeds_list] + [tax], axis=1)

    return embeds, fasta['EntryID'].values


def get_targets_from_parquet(path, ontologies, split, ids=None, names=None, fillna=False):
    res = []

    for i in range(3):

        if split[i] == 0:
            continue

        G = ontologies[i]
        start = sum(split[:i])
        stop = start + split[i]

        if ids is not None:
            names_ = get_funcs_mapper(G, False)[ids[start: stop]].tolist()
        elif names is not None:
            names_ = list(names[start: stop])
        else:
            raise ValueError()

        flist = sorted(glob.glob(os.path.join(path, G.namespace, 'part*')))
        trg = pd.concat([
            pd.read_parquet(x, columns=names_) for x in flist
        ], ignore_index=True)  # pd.read_parquet(flist, columns=names_)

        if fillna:
            print('trg filled')
            trg = trg.fillna(0)

        res.append(trg.values)

    return np.concatenate(res, axis=1)

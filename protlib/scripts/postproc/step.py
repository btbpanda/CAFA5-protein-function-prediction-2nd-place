import argparse
import os
import sys

import tqdm
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)

parser.add_argument('-d', '--device', type=str, default="1")
parser.add_argument('-b', '--batch_size', type=int, default=30000)
parser.add_argument('-bi', '--batch_inner', type=int, default=5000)
parser.add_argument('-l', '--lr', type=float, default=0.1)
parser.add_argument('-dr', '--direction', type=str, default='max')


def get_kernel(direction):
    kernel = cp.ElementwiseKernel(
        """
        int64 i_,
        raw int64 j_,
        int64 col,
        int64 l,
        int64 lx
        """,
        'raw float32 arr',
        """
        int k;
    
        for (k = 0; k < l; k++) {{
            arr[i_ * lx + col] = {0}(arr[i_ * lx + col], arr[i_ * lx + j_[k]]);
        }}
        """.format(direction),
        'propagate_{0}_kernel'.format(direction)
    )

    return kernel


def propagate_max(mat, G):
    indexer = cp.arange(mat.shape[0])

    for f in G.order:

        adj = G.terms_list[f]['children']

        if len(adj) == 0:
            continue

        adj = cp.asarray(adj, dtype=cp.int64)
        prop_max_kernel(indexer, adj, f, adj.shape[0], mat.shape[1], mat.ravel())

    return


def propagate_min(mat, G):
    indexer = cp.arange(mat.shape[0])

    D = get_depths(G, True)
    for i in range(len(D)):
        for f in D[i]:

            adj = G.terms_list[f]['adj']

            if len(adj) == 0:
                continue

            adj = cp.asarray(adj, dtype=cp.int64)
            prop_min_kernel(indexer, adj, f, adj.shape[0], mat.shape[1], mat.ravel())

    return


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import cupy as cp
    import cudf

    try:
        from protlib.metric import get_funcs_mapper, get_ns_id, obo_parser, Graph, get_depths
    except Exception:
        get_funcs_mapper, get_ns_id, obo_parser, Graph = [None] * 4

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    prop_max_kernel = get_kernel('max')
    prop_min_kernel = get_kernel('min')

    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    pp_path = os.path.join(config['base_path'], config['models_path'], 'postproc')
    input_path = os.path.join(pp_path, 'pred.tsv')
    output_path = os.path.join(pp_path, f'pred_{args.direction}.tsv')

    trainTerms = cudf.read_csv(input_path, sep='\t', names=['EntryID', 'term', 'prob'], header=None)
    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    back_prot_id = trainTerms['EntryID'].drop_duplicates().reset_index(drop=True)
    length = len(back_prot_id)
    prot_id = cudf.Series(cp.arange(length), back_prot_id)
    trainTerms['id'] = trainTerms['EntryID'].map(prot_id)

    flg = True

    for i in tqdm.tqdm(range(0, length, args.batch_size)):

        sample = trainTerms.query(f'(id >= {i}) & (id < {i + args.batch_size})')
        batch_len = min(args.batch_size, length - i)

        for G in ontologies:
            mapper = cudf.Series(get_funcs_mapper(G))
            sample['term_id'] = sample['term'].map(mapper)
            sample_ont = sample.dropna().astype({'term_id': cp.int32})

            sample_ont['id'] = sample_ont['id'] - i

            mat = cp.zeros((batch_len, G.idxs), dtype=cp.float32)
            mat.scatter_add((sample_ont['id'].values, sample_ont['term_id'].values), sample_ont['prob'].values)
            mat = cp.clip(mat, 0, 1)
            mat_old = mat.copy()

            if args.direction == 'max':
                propagate_max(mat, G)
                # mat = cp.maximum(mat, mat_old)
            else:
                propagate_min(mat, G)
                # mat = cp.minimum(mat, mat_old)

            mat = mat * args.lr + mat_old * (1 - args.lr)

            for j in range(0, mat.shape[0], args.batch_inner):
                mat_batch = mat[j: j + args.batch_inner]
                row, col = cp.nonzero(mat_batch)

                sample_batch = cudf.DataFrame({

                    'EntryID': back_prot_id[i + j + row].reset_index(drop=True),
                    'term': cudf.Series(get_funcs_mapper(G, False))[col].reset_index(drop=True),
                    'prob': mat_batch[row, col]
                }).sort_values(['EntryID', 'term'], ascending=True)

                sample_batch['prob'] = sample_batch['prob'].astype(str).str.slice(0, 5)

                ns_id, ns_str = get_ns_id(G)
                asp = ns_str.upper() + 'O'

                with open(output_path, 'w' if flg else 'a') as f:
                    sample_batch.to_csv(f, index=False, sep='\t', header=None)
                    flg = False

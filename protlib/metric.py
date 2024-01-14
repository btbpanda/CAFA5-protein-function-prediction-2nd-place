import tqdm

try:
    import cudf
except ImportError:
    cudf = None

try:
    import cupy as cp
except ImportError:
    cp = None

from .cafa_utils import *


def get_target(trainTerms, G):
    ns_id, ns_str = get_ns_id(G)
    asp = ns_str.upper() + 'O'

    sample = trainTerms.query(f'aspect == "{asp}"').copy()
    sample['ID'] = sample['term'].map(get_funcs_mapper(G))
    sample['gt'] = np.ones(sample.shape[0], dtype=cp.float32)

    return cudf.from_pandas(sample.drop(['term', 'aspect'], axis=1).rename(columns={'EntryID': 'entry_id'}))


def get_ia(G, ia_path):
    ia_dict = ia_parser(ia_path)
    return get_funcs_mapper(G, False).map(ia_dict)


def get_topk_targets(G, topk, train_path='Train', trainTerms=None, ex_top=False, freq_co=0):
    if trainTerms is None:
        trainTerms = pd.read_csv(f'{train_path}/train_terms.tsv', sep='\t', usecols=['term', 'aspect'])
    ns_id, ns_str = get_ns_id(G)

    asp = ns_str.upper() + 'O'
    sample = trainTerms.query(f'aspect == "{asp}"').copy()
    sample['id'] = sample['term'].map(get_funcs_mapper(G)).values

    vc = sample.groupby(['term', 'id']).size()
    vc = vc[vc >= freq_co]
    vc = vc.sort_values(ascending=False)[int(ex_top):topk].reset_index()

    return vc['id'].tolist()


def get_funcs_mapper(G, fwd=True):
    mapper = [x['id'] for x in G.terms_list]
    if fwd:
        return pd.Series(np.arange(len(mapper)), index=mapper)

    return pd.Series(mapper)


def get_ns_id(G):
    ns_id = ['biological_process', 'molecular_function', 'cellular_component'].index(G.namespace)
    ns_str = ''.join(map(lambda x: x[0], G.namespace.split('_')))

    return ns_id, ns_str


def get_depths(G, top=False):
    D = {}

    d = 0
    nodes = [(n, x) for (n, x) in enumerate(G.terms_list) if len(x['adj']) == 0]

    if top:
        D[nodes[0][0]] = 0
        d = 1

    nodes = [x[1] for x in nodes]

    while len(nodes) > 0:

        new_nodes = []

        for n in nodes:
            for k in n['children']:
                new_nodes.append(k)
                D[k] = d

        new_nodes = [G.terms_list[x] for x in set(new_nodes)]

        nodes = new_nodes
        d += 1

    return pd.Series(D).reset_index().groupby(0)['index'].agg(list).to_dict()


propagate_col_kernel = cp.ElementwiseKernel(
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

    for (k = 0; k < l; k++) {
        arr[i_ * lx + col] = max(arr[i_ * lx + col], arr[i_ * lx + j_[k]]);
    }

    """,

    'propagate_col_kernel') if cp is not None else None

conditional_col_kernel = cp.ElementwiseKernel(
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
    float acc = 1;

    for (k = 0; k < l; k++) {
        acc = acc *  (1 - arr[i_ * lx + j_[k]]);
    }

    arr[i_ * lx + col] = arr[i_ * lx + col] * (1 - acc)

    """,

    'conditional_col_kernel') if cp is not None else None


def fix_conditional(mat, G, cond_idx=None):
    D = get_depths(G)
    indexer = cp.arange(mat.shape[0], dtype=cp.int64)
    for i in range(len(D)):
        for k in D[i]:
            if cond_idx is None or k in cond_idx:
                adj = cp.asarray(G.terms_list[k]['adj'], dtype=cp.int64)
                conditional_col_kernel(indexer, adj, k, adj.shape[0], mat.shape[1], mat.ravel())


def propagate_correct(mat, G, mode='fill'):
    for f in G.order:

        adj = G.terms_list[f]['children']

        if len(adj) == 0:
            continue

        indexer = cp.where(cp.ascontiguousarray(mat[:, f]) == 0)[0]

        if len(indexer) == 0:
            continue

        adj = cp.asarray(adj, dtype=cp.int64)
        propagate_col_kernel(indexer, adj, f, adj.shape[0], mat.shape[1], mat.ravel())

    return


def propagate_cafa(mat, G):
    for f in G.order:

        adj = G.terms_list[f]['children']

        if len(adj) == 0:
            continue

        indexer = cp.where(cp.ascontiguousarray(mat[:, f]) == 0)[0]

        if len(indexer) == 0:
            continue

        adj = cp.asarray(adj, dtype=cp.int64)
        fill_value = mat[indexer[0], adj].max()
        mat[indexer, f] = fill_value

    return


def propagate(mat, G, mode='fill'):
    if mode == 'fill':
        propagate_correct(mat, G)
    elif mode == 'cafa':
        propagate_cafa(mat, G)

    return


def propagate_df(batch, G, n_funcs, mode='fill'):
    if mode not in ['fill', 'cafa']:
        return batch

    rows = batch['entry_num'].values
    cols = batch['ID'].values
    probs = batch['prob'].values
    # print(rows)
    mat = cp.zeros((int(rows[-1]) + 1, n_funcs), dtype=cp.float32)
    mat.scatter_add((rows, cols), probs)

    propagate(mat, G, mode)

    row, col = cp.nonzero(mat)
    val = mat[row, col]

    batch = cudf.DataFrame({

        'entry_num': row,
        'ID': col,
        'prob': val
    }).sort_values(['entry_num', 'prob'], ascending=[True, False])

    return batch


def iterate_from_df(df, G, batch_size, idx, back_idx, prop_mode='fill'):
    # if filename - read
    if type(df) is str:
        sub = cudf.read_csv(df, sep='\t', header=None, names=['entry_id', 'ID', 'prob'])
    else:
        sub = df
    n_funcs = len(G.terms_list)

    check = cudf.Series(cp.ones(n_funcs, dtype=np.float32), index=[x['id'] for x in G.terms_list])
    sub_single = sub[sub['ID'].map(check).notnull()]
    # add numeric id for function inside the domain

    fmap = cudf.from_pandas(get_funcs_mapper(G))
    sub_single['ID'] = sub_single['ID'].map(fmap)

    sub_single['entry_num'] = sub_single['entry_id'].map(idx)
    sub_single = sub_single.sort_values('entry_num')

    # propagate and iterate
    nrows = int(sub_single['entry_num'].max() + 1)

    for i in tqdm.tqdm(range(0, nrows, batch_size)):
        nrows_batch = min(batch_size, nrows - i)
        batch = sub_single.query(f'(entry_num >= {i}) & (entry_num < {i + nrows_batch})')
        batch['entry_num'] = batch['entry_num'] - i
        # propagate
        batch = propagate_df(batch, G, n_funcs, mode=prop_mode)
        batch['entry_num'] = batch['entry_num'] + i
        batch['entry_id'] = batch['entry_num'].map(back_idx)

        # print(batch.shape)

        yield batch

def aggregate_fn(df, col, n_un, n_bins):
    pvt = cp.zeros((n_un, n_bins), dtype=np.float32)
    pvt.scatter_add((df['temp_id'].values, df['bin'].values), df[col].values)

    pvt = pvt.sum(axis=1, keepdims=True) - pvt.cumsum(axis=1)

    return pvt


class CAFAMetric:

    def __init__(self, obo_path, ia_path, helpers_path, prop_mode='fill', batch_size=10000, tau=0.01, topk=500):

        self.ia_dict = ia_parser(ia_path)
        self.ia_path = ia_path
        self.ontologies = []

        for ns, terms_dict in obo_parser(obo_path).items():
            self.ontologies.append(Graph(ns, terms_dict, self.ia_dict, True))

        self.helpers_path = helpers_path
        self.prop_mode = prop_mode
        self.batch_size = batch_size
        self.tau = tau
        self.topk = topk

    def calc_stats(self, batch, target, unique, G):

        n_bins = 1001

        unique = cudf.DataFrame(unique).rename(columns={0: 'entry_id'})
        unique['temp_id'] = cp.arange(unique.shape[0], )
        unique = unique.set_index('entry_id')['temp_id']

        n_un = unique.shape[0]

        # add gt
        batch['pred'] = cp.ones(batch.shape[0], dtype=cp.float32)
        merged = cudf.merge(batch, target, on=['entry_id', 'ID'], how='outer').fillna(0)
        # filter toi
        toi_sl = merged['ID'].isin(cp.asarray(G.toi))  # .astype(cp.float32)
        merged['temp_id'] = merged['entry_id'].map(unique)

        # weights
        merged['ia'] = merged['ID'].map(cudf.from_pandas(get_ia(G, self.ia_path))).astype(cp.float32)

        merged['bin'] = cp.floor(merged['prob'] * 1000).astype(cp.int32)
        # weighted stats
        merged['inter'] = (merged['pred'] == merged['gt']).astype(np.float32) * merged['ia']
        merged['wgt'] = merged['gt'] * merged['ia']
        merged['wpred'] = merged['pred'] * merged['ia']

        merged['flg'] = merged['wpred'] > 0

        # calc cov
        merged['bin_x'] = merged['bin'] + 1
        mtoi = merged.query('flg')

        cov = mtoi['entry_id'].nunique() - mtoi \
            .groupby('entry_id')['bin_x'].max() \
            .value_counts() \
            .to_frame() \
            .join(cudf.DataFrame([], index=cudf.RangeIndex(n_bins)), how='right') \
            .sort_index() \
            .fillna(0) \
            .cumsum()['bin_x'].values

        inter = aggregate_fn(merged, 'inter', n_un, n_bins)
        pred = aggregate_fn(merged, 'wpred', n_un, n_bins)

        gt = merged.groupby('temp_id')['wgt'].sum().sort_index().values[:, cp.newaxis]

        pr = np.where(pred == 0, 0, inter / pred).sum(axis=0)
        rc = np.where(gt == 0, 0, inter / gt).sum(axis=0)

        return pr, rc, cov


    def from_df(self, y_true, y_pred):

        if type(y_true) is str:
            y_true = pd.read_csv(y_true, sep='\t')

        if type(y_pred) is str:
            y_pred = cudf.read_csv(y_pred, sep='\t', header=None, names=['entry_id', 'ID', 'prob'])

        assert (np.array(y_pred.columns) == np.array(['entry_id', 'ID', 'prob'])).all()

        metrics = {}

        for G, name in zip(self.ontologies, ['bp', 'mf', 'cc']):
            target = get_target(y_true, G)
            if len(target) == 0:
                continue

            ent = target[['entry_id']].drop_duplicates() \
                .sort_index() \
                .reset_index(drop=True) \
                .reset_index()

            ns_pred = cudf.merge(y_pred, ent, on='entry_id')

            metrics[name] = self.from_df_single(
                target, ns_pred, G, ent
            )

        if all([x in metrics for x in ['bp', 'mf', 'cc']]):
            metrics['cafa'] = np.mean(list(metrics.values()))

        return metrics

    def from_df_single(self, y_true, y_pred, G, idx):

        pr, rc, cov = None, None, None

        back_idx = idx.set_index('index')['entry_id']
        idx = idx.set_index('entry_id')['index']

        for num, batch in tqdm.tqdm(enumerate(iterate_from_df(
                y_pred, G, self.batch_size, idx, back_idx, prop_mode=self.prop_mode
        ))):
            unique = back_idx[num * self.batch_size: (num + 1) * self.batch_size]
            trg = y_true[y_true['entry_id'].isin(unique)]
            pr, rc, cov = [
                (x if y is None else x + y) for (x, y) in zip(
                    self.calc_stats(batch, trg, unique, G), [pr, rc, cov]
                )
            ]

        pr = pr / cov
        rc = rc / len(back_idx)
        f1 = 2 * pr * rc / (pr + rc)

        return float(cp.nanmax(f1))

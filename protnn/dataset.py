import numpy as np
import torch
from numba import jit
from torch.utils.data import Dataset, DataLoader

try:
    from protlib.metric import get_depths
    import cupy as cp
except ImportError:
    cp, get_depths = [None] * 2


@jit(nopython=True)
def propagate(arr, cnd_mask, col, n_mod, adj):
    for j in range(n_mod):

        if not cnd_mask[j, col]:
            continue

        acc_p = 1.0
        acc_1mp = 1.0

        for k in adj:
            acc_1mp *= 1 - arr[j * 4 + 3, k]
            acc_p *= arr[j * 4 + 2, k]

        arr[j * 4 + 3, col] *= 1 - acc_1mp
        arr[j * 4 + 2, col] *= acc_p

    return


class Propagator:

    def __init__(self, G, preds):

        self.G = G
        self.D = get_depths(G)
        cnd_mask = np.ones((len(preds), G.idxs), dtype=np.bool_)

        for n, (_, idx, cond) in enumerate(preds):
            if cond:
                continue
            cnd_mask[n, idx] = False

        self.adj = [np.asarray(x['adj']) for x in G.terms_list]

        self.cnd_mask = np.asarray(cnd_mask)
        self.n_mod = len(preds)

    def __call__(self, batch):

        arr = batch['x'].numpy()

        for i in range(len(self.D)):
            for k in self.D[i]:
                # for k in self.G.order:
                adj = self.adj[k]
                if len(adj) == 0:
                    continue

                propagate(arr, self.cnd_mask, k, self.n_mod, adj)

        return batch


def get_dag_dense(G, direction='all', self_loop=True):
    dst, src = [], []

    for i, node in enumerate(G.terms_list):
        if self_loop:
            dst.append(i)
            src.append(i)

        if direction in ['all', 'fwd']:
            for j in node['adj']:
                dst.append(i)
                src.append(j)

        if direction in ['all', 'bwd']:
            for j in node['children']:
                dst.append(i)
                src.append(j)

    dst, src = torch.LongTensor(dst), torch.LongTensor(src)

    return dst, src


class StackDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mod = len(self.dataset.preds)
        self.lo_idx = torch.arange(self.n_mod * 4).reshape((self.n_mod, 4))[:, 1:].ravel().cuda()

        for direction in ['all', 'fwd', 'bwd']:
            dst, src = get_dag_dense(self.dataset.G, direction=direction, self_loop=False)
            self.__dict__[direction] = {'dst': dst.cuda(), 'src': src.cuda()}

    def __iter__(self, ):
        for batch in super().__iter__():
            batch = {x: batch[x].cuda() for x in batch}
            arr = torch.clamp(batch['x'][:, self.lo_idx], 1e-6, 1 - 1e-6)
            batch['x'][:, self.lo_idx] = torch.log(arr / (1 - arr))

            batch['x'] = batch['x'].swapaxes(1, 2)
            batch['x'], batch['goa'] = batch['x'][..., :self.n_mod * 4], batch['x'][..., self.n_mod * 4:]

            for direction in ['all', 'fwd', 'bwd']:
                batch[direction] = self.__dict__[direction]

            yield batch


class StackDataset(Dataset):

    def __init__(self, preds, nout, prior_raw, prior_cond, G, goa_list, p_goa=1, targets=None):

        self.preds = preds
        self.nout = nout
        self.prior_raw = prior_raw
        self.prior_cond = prior_cond
        self.G = G
        self.goa = [x.tolist() for x in goa_list]
        self.p_goa = p_goa

        self.targets = targets
        self.adj = [np.array(x['adj'], dtype=np.int64) for x in G.terms_list]
        self.prop = Propagator(G, preds)

    def __getitem__(self, index):

        batch = {}
        x = []

        for n, (pred, idx, cond) in enumerate(self.preds):
            arr = np.ones((4, self.nout), dtype=np.float32)
            arr[0, idx] = 0  # indicator that prediction comes from prior
            arr[1] = self.prior_cond if cond else self.prior_raw  # prior for raw prediction
            arr[2:] = self.prior_cond  # prior for propagated prediction
            # (assume one of parents for 2 index and assume all parents for 3 index)
            arr[1:, idx] = pred[index]  # fill with known predictions
            x.append(torch.from_numpy(arr))

        # add go annotations
        goa = np.zeros((len(self.goa), self.nout), dtype=np.float32)
        # if np.random.rand() < self.p_goa:
        for n, ann in enumerate(self.goa):
            ann = ann[index]
            if len(ann) > 0:
                goa[n, ann] = 1
        x.append(torch.from_numpy(goa))

        batch['x'] = torch.cat(x, dim=0)
        self.prop(batch)

        if self.targets is not None:
            batch['y'] = torch.from_numpy(self.targets[index])

        return batch

    def __len__(self, ):

        return self.preds[0][0].shape[0]

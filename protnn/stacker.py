import torch
from torch import nn


class GCNLayer(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        self.w0 = nn.Linear(in_features * 2, in_features)
        self.act0 = nn.ReLU()
        self.w1 = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x0, dst, src):
        x = torch.cat([
            x0.index_reduce(1, dst, x0[:, src], reduce='mean', include_self=False),
            x0.index_reduce(1, dst, x0[:, src], reduce='amax', include_self=False)
        ], dim=2)

        x = self.w0(x)
        x = self.act0(x)
        x = self.dropout(x)
        x = self.w1(x)

        x = x0 + x

        return x


class GCNStacker(nn.Module):

    def __init__(self, in_models, in_goa, out_features, hidden_size=16, n_layers=8, embed_size=16):

        super().__init__()

        self.in_models = in_models
        self.in_goa = in_goa
        # self.in_features = in_models * 4 + in_goa

        self.nout = out_features
        self.n_layers = n_layers
        self.embed_size = embed_size
        # node embedding

        if embed_size > 0:
            self.node_embed = nn.Embedding(self.nout, self.embed_size)

        node_feats = hidden_size + embed_size

        self.input = nn.Linear(self.in_models * 4 + in_goa, hidden_size)
        self.act = nn.ReLU()
        self.bn0 = nn.LayerNorm(in_models * 4)
        # self.dropout = nn.Dropout1d(0.5)
        # self.dropout = nn.Dropout1d(0.5)

        self.gcn_alldir = nn.ModuleList()
        for i in range(self.n_layers):
            self.gcn_alldir.append(GCNLayer(node_feats))

        self.gcn_fwd = nn.ModuleList()
        for i in range(self.n_layers):
            self.gcn_fwd.append(GCNLayer(node_feats))

        self.gcn_bwd = nn.ModuleList()
        for i in range(self.n_layers):
            self.gcn_bwd.append(GCNLayer(node_feats))

        self.clf = nn.Linear(node_feats * 4, 1)
        # self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)

    def forward(self, batch):

        l = batch['x'].shape[0]

        x = self.bn0(batch['x'])
        goa = batch['goa']
        # goa = self.dropout(batch['goa'])
        # goa = batch['goa']
        x = torch.cat([x, goa], dim=2)
        x = self.input(x)
        x = self.act(x)

        if self.embed_size > 0:
            emb = self.node_embed(torch.tile(torch.arange(self.nout).cuda(), (l, 1)))
            x = torch.cat([x, emb], dim=2)

        # x = self.bn0(x)
        # n_samples * n_nodes * n_features

        x0 = x  # n_samples * n_features * n_nodes

        layers = [x0]

        for gcns, direction in zip(
                [self.gcn_alldir, self.gcn_bwd, self.gcn_fwd],
                ['all', 'bwd', 'fwd']
        ):
            x = x0
            for gcn in gcns:
                x = gcn(x, **batch[direction])
            layers.append(x)

        x = torch.cat(layers, dim=2)  # n_samples * n_features * n_nodes

        x = self.clf(x)[..., 0]  # + self.bias  # n_samples * n_nodes * n_features
        return x

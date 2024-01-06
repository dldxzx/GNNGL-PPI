import torch
from torch_geometric.nn import GINConv
from torch_geometric.nn.inits import reset
from torch_scatter import scatter

from elements import *


# class BiLinearAttention(nn.Module):
#     def __init__(self, input_size):
#         super(BiLinearAttention, self).__init__()
#         self.weight_matrix = nn.Parameter(torch.rand(input_size, input_size))
#
#     def forward(self, x1, x2):
#         # x1, x2: (batch_size, input_size)
#
#         # 计算注意力权重
#         attention_weights = torch.matmul(x1, self.weight_matrix)
#         attention_weights = torch.matmul(attention_weights, x2.t())
#
#         # 使用 softmax 归一化
#         attention_weights = torch.softmax(attention_weights, dim=-1)
#
#         # 使用注意力权重加权得到加权和
#         weighted_sum = torch.matmul(attention_weights, x2)
#
#         return weighted_sum, attention_weights


class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nin, nout, nlayer, gnn_type, dropout=0, bn=True, bias=True, res=True):
        super().__init__()
        # TODO: consider remove input and output encoder for nhead=1?
        # self.input_encoder = MLP(nin, nin, nlayer=2, with_final_activation=True) #if nin!=nout else nn.Identity()
        self.convs = GINConv(
            nn.Sequential(
                nn.Linear(512 + 16, nin),
            ), train_eps=True
        )  # set bias=False for BN
        self.norms = nn.BatchNorm1d(nin) if bn else Identity()
        self.output_encoder = MLP(nin, nout, nlayer=1, with_final_activation=False,
                                  bias=bias) if nin != nout else Identity()
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        # self.input_encoder.reset_parameters()
        self.output_encoder.reset_parameters()
        self.convs.reset_parameters()
        self.norms.reset_parameters()

    def forward(self, x, edge_index,  batch):
        # x = self.input_encoder(x)
        previous_x = x
        x = self.convs(x, edge_index)
        x = self.norms(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.res:
            x = x + previous_x
            previous_x = x

        x = self.output_encoder(x)

        return x


class SubgraphGNNKernel(nn.Module):
    # Use GNN to encode subgraphs
    # gnn_types: a list of GNN types
    def __init__(self, nin, nout, nlayer, gnn_types, dropout=0,
                 hop_dim=16,
                 bias=True,
                 res=True,
                 pooling='mean',
                 embs=(0, 1, 2),
                 embs_combine_mode='add',
                 mlp_layers=1,
                 subsampling=False,
                 online=True):
        super().__init__()
        assert max(embs) <= 2 and min(embs) >= 0
        assert embs_combine_mode in ['add', 'concat']

        use_hops = hop_dim > 0
        nhid = 512
        self.hop_embedder = nn.Embedding(20, hop_dim)

        self.gnns = GNN(nin + hop_dim if use_hops else nin, nhid, nlayer, gnn_types, dropout=dropout, res=res)

        self.subgraph_transform = MLP(nout, nout, nlayer=mlp_layers, with_final_activation=True)
        self.context_transform = MLP(nout, nout, nlayer=mlp_layers, with_final_activation=True)

        self.out_encoder = MLP(nout if embs_combine_mode == 'add' else nout * len(embs), nout, nlayer=mlp_layers,
                               with_final_activation=False, bias=bias, with_norm=True)

        self.use_hops = use_hops
        self.gate_mapper_subgraph = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_context = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_centroid = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        # add this one to scale centroid embedding
        self.subsampling = subsampling

        # dropout = 0
        self.dropout = dropout
        self.online = online
        self.pooling = pooling
        self.embs = embs
        self.embs_combine_mode = embs_combine_mode


    def reset_parameters(self):
        self.hop_embedder.reset_parameters()

        self.gnns.reset_parameters()
        self.subgraph_transform.reset_parameters()
        self.context_transform.reset_parameters()
        self.out_encoder.reset_parameters()
        reset(self.gate_mapper_context)
        reset(self.gate_mapper_subgraph)
        reset(self.gate_mapper_centroid)

    def forward(self, data):



        # prepare x, edge_index, edge_attr for the combined subgraphs
        combined_subgraphs_x = data.x[data.subgraphs_nodes_mapper]  # lift up the embeddings, positional encoding


        combined_subgraphs_edge_index = data.combined_subgraphs
        # combined_subgraphs_edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        combined_subgraphs_batch = data.subgraphs_batch

        if self.use_hops:
            hop_emb = self.hop_embedder(
                data.hop_indicator + 1)  # +1 to make -1(not calculated part: too far away) as 0.
            combined_subgraphs_x = torch.cat([combined_subgraphs_x, hop_emb], dim=-1)
        # print(combined_subgraphs_x.shape)

        combined_subgraphs_x = self.gnns(combined_subgraphs_x, combined_subgraphs_edge_index,
                                          combined_subgraphs_batch)

        centroid_x = combined_subgraphs_x[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)]
        subgraph_x = self.subgraph_transform(
            F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(
            self.embs) > 1 else combined_subgraphs_x
        context_x = self.context_transform(
            F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(
            self.embs) > 1 else combined_subgraphs_x
        if self.use_hops:
            centroid_x = centroid_x * self.gate_mapper_centroid(
                hop_emb[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)])
            subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_emb)
            context_x = context_x * self.gate_mapper_context(hop_emb)
        subgraph_x = scatter(subgraph_x, combined_subgraphs_batch, dim=0, reduce=self.pooling)
        context_x = scatter(context_x, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)
        # print(centroid_x.shape)
        # print(subgraph_x.shape)
        # print(context_x.shape)

        x = [centroid_x, subgraph_x, context_x]
        x = [x[i] for i in self.embs]



        if self.embs_combine_mode == 'add':
            # print('================embs_combine_mode=add===================')
            x = sum(x)
        else:
            x = torch.cat(x, dim=-1)
            # last part is only essential for embs_combine_mode = 'concat', can be ignored when overfitting
            # x = self.out_encoder(F.dropout(x, self.dropout, training=self.training))


        return x

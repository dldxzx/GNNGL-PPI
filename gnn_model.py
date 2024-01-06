import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge

from GIn import GINConv
# from transformer import GotermEncoder

# from CSRA_ import MHA
from subgraph_model import SubgraphGNNKernel



class GNNGL_PPI(torch.nn.Module):
    def __init__(self, graph, gin_in_feature=256, num_layers=1,
                 hidden=512, use_jk=False, train_eps=True,
                 feature_fusion=None, class_num=7, ):
        super(GNNGL_PPI, self).__init__()
        self.graph = graph
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion
        # self.fc_dy = nn.Linear(512, 512)
        # self.uttransformer = UT_EncoderLayer(666, 666, 666, 666, num_heads=2)
        # self.transformer = EncoderLayer(d_model=666, d_inner=666, n_head=8, d_k=13, d_v=13)

        self.fc_x = nn.Linear(512, 512)
        #
        # self.conv1d = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=3, padding=0)
        # self.bn1 = nn.BatchNorm1d(1)
        # self.biGRU = nn.GRU(1, 1, bidirectional=True, batch_first=True, num_layers=1)
        # self.maxpool1d = nn.MaxPool1d(3, stride=3)
        # self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        # self.fc1 = nn.Linear(math.floor(2000 / 3), 512)
        # self.fc_a = nn.Linear(512, 7)

        self.subgraph_layers = SubgraphGNNKernel(512, 512, 1, 'GINConv', 0.2,
                                                 hop_dim=16,
                                                 bias=False,
                                                 res=True,
                                                 pooling='mean',
                                                 embs=(0, 1),
                                                 embs_combine_mode='add',
                                                 mlp_layers=1,
                                                 subsampling=False, )
        self.norm = nn.BatchNorm1d(512)

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(512, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(512, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )


        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=self.train_eps
                )
            )
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)
        self.lin1_o = nn.Linear(hidden, hidden)
        self.lin2_o = nn.Linear(hidden, hidden)


    def reset_parameters(self):

        self.conv1d.reset_parameters()
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()

        self.gin_conv2.reset_parameters()

        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()

        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.fc2.reset_parameters()

    def forward(self, x, edge_index, train_edge_id, edge_go, graph, p=0.5):

        sub_x = self.subgraph_layers(graph)
        sub_x = self.norm(sub_x)
        sub_x = F.relu(sub_x)
        sub_x = F.dropout(sub_x, 0.5, training=self.training)

        x = self.fc_x(x)
        x = self.gin_conv1(x, edge_index)

        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs += [x]

        if self.use_jk:
            x = self.jump(xs)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin2(x)

        x = x + sub_x

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)

        x = self.fc2(x)



        return x

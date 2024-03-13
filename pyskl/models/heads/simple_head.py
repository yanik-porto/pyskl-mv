import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead

from torch_geometric.nn import GCNConv, GraphConv, Linear, global_mean_pool

class AttentionMV(torch.nn.Module):
    def __init__(self, feature_size):
        super(AttentionMV, self).__init__()

        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, 120)

    def forward(self, x1, x2, mask=None):
        queries = self.query(x1)
        keys = self.key(x2)
        values = self.value(x2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights


class GNNMV(torch.nn.Module):
    def __init__(self, in_features, hidden_features=16, out_features=60):
        super(GNNMV, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.conv3 = GCNConv(hidden_features, out_features)
        # self.lin = Linear(hidden_features, out_features)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        # print(x)
        # x = x.relu()
        x = F.relu(x)
        # print(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = x.relu()
        # x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        # x = x.reshape(-1, 6, x.shape[-1])
       # print("x after : ", x.shape)
        # x = x.mean(dim=1)
        # x = x.squeeze(dim=1)

        # print(x)

        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        x = self.conv3(x, edge_index)
        
        return x

class GCNMV(nn.Module):
    def __init__(self, in_features, hidden_features=16, out_features=60):
        super().__init__()
        self.conv1 = GraphConv(in_features, hidden_features)
        self.conv1_bis = GraphConv(hidden_features, hidden_features)
        self.conv2 = GraphConv(hidden_features, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)

        x = F.tanh(x)

        x = self.conv1_bis(x, edge_index)
        x = F.tanh(x)

        x= F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Tanh, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.A = torch.tensor([ [0., 0., 1., 0., 1., 0.],
                                [0., 0., 0., 1., 0., 1.],
                                [1., 0., 0., 0., 1., 0.],
                                [0., 1., 0., 0., 0., 1.],
                                [1., 0., 1., 0., 0., 0.],
                                [0., 1., 0., 1., 0., 0.]])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x

@HEADS.register_module()
class SimpleHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D', 'GCNMV', 'GNNMV', 'ATTNMV']
        self.mode = mode

        self.in_c = in_channels
        # self.fc_cls = nn.Linear(self.in_c, num_classes)

        n_views = 2
        n_persons = 2
        self.node_number = n_views * n_persons

        hidden_features = 2048
        # self.mlp_mv = nn.Linear(self.in_c * n_views * n_persons, self.in_c)
        # self.mlp_mv = MLP(self.in_c * n_views * n_persons, hidden_features=2048, out_features=self.in_c)
        # self.fc_cls = nn.Linear(hidden_features, num_classes)
        # self.fc_cls = nn.Linear(self.in_c, num_classes)



        if self.mode == "GCNMV":
            self.gcn_mv = GCNMV(self.in_c, hidden_features=hidden_features, out_features=num_classes)
        elif self.mode == "GNNMV":
            self.gcn_mv = GNNMV(self.in_c, hidden_features=hidden_features, out_features=num_classes)
        elif self.mode == "ATTNMV":
            self.attn = AttentionMV(self.in_c)
            self.drop_attn = nn.Dropout(0.3)
        else:
            self.fc_cls = nn.Linear(self.in_c, num_classes)
            # self.fc_cls_attn = nn.Linear(self.in_c, num_classes)
        # self.edge_index = torch.tensor([[0, 2],
        #                                 [2, 0],
        #                                 [0, 4],
        #                                 [4, 0],
        #                                 [1, 3],
        #                                 [3, 1],
        #                                 [1, 5],
        #                                 [5, 1],
        #                                 [2, 4],
        #                                 [4, 2],
        #                                 [3, 5],
        #                                 [5, 3]], dtype=torch.long)


        self.edge_index = torch.tensor([[0, 2],
                                        [2, 0],
                                        [1, 3],
                                        [3, 1]], dtype=torch.long)


        self.batch_idxs = torch.tensor([0] * self.node_number, dtype=torch.long)

    def get_edge_indexes(self, N):
        edge_index = self.edge_index.clone()
        # print("edge_index size : ", edge_index.shape)

        for i in range(1, N):
            newedges = self.edge_index.clone() + i * self.node_number
            # print("newedges: ", newedges)
            edge_index = torch.cat((edge_index, newedges), dim=0)
            # print("edge_index: ", edge_index)

        # print("edge_index size : ", edge_index.shape)
        return edge_index.t().contiguous().cuda()

    def get_batch_indexes(self, N):
        batch_idxs = self.batch_idxs.clone()

        for i in range(1, N):
            newbatchs = self.batch_idxs.clone() + i
            batch_idxs = torch.cat((batch_idxs, newbatchs), dim=0)
        
        # print("batch_idxs: ", batch_idxs)
        return batch_idxs.cuda()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        if not (self.mode == "GCNMV" or self.mode == "GNNMV" or self.mode == "ATTNMV"):
            normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)
            if self.mode == 'GCNMV':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)
                x = pool(x)
                # x = x.reshape(N, M * C)
                # x = self.mlp_mv(x)

                # x = x.reshape(N, M, C)
                x = x.reshape(N * M, C)
                # print("N : ", N)
                # print("N : ", M)
                # print("N : ", C)
                # print("x before : ", x.shape)
                # x = self.gcn_mv(x, self.edge_index * 10)
                # x = self.gcn_mv(x, self.edge_index)
                x = self.gcn_mv(x, self.get_edge_indexes(N))

                # print(x)
                # print(x.shape)
                # print("x after linear: ", x.shape)

                x = x.reshape(N, M, -1)

                # print("x after : ", x.shape)

                x = x.mean(dim=1)
                x = x.squeeze(dim=1)

                # x = x.reshape(N, -1)
                # print("x after reshape : ", x.shape)
            if self.mode == 'GNNMV':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)
                x = pool(x)
                x = x.reshape(N * M, C)
                x = self.gcn_mv(x, self.get_edge_indexes(N), self.get_batch_indexes(N))

            if self.mode == 'ATTNMV':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)
                x = pool(x)
                x = x.reshape(N, M, C)

                # print("before attn: ", x.shape)
                # x , _ = self.attn(x[:, 0:2, :], x[:, 2:4, :])
                x , _ = self.attn(x[:, 0, :], x[:, 2, :])
                # x , _ = self.attn(x)

                # print("after attn: ", x.shape)
                # x = self.drop_attn(x)
                # print("after drop: ", x.shape)

                # x = self.fc_cls_attn(x)
                # print("after cls: ", x.shape)

                # x = x.mean(dim=1)
                # x = x.squeeze(dim=1)




        if self.mode == "GCNMV" or self.mode == "GNNMV" or self.mode == "ATTNMV":
            cls_score = x
        else:
            assert x.shape[1] == self.in_c
            if self.dropout is not None:
                x = self.dropout(x)

            cls_score = self.fc_cls(x)
        return cls_score


@HEADS.register_module()
class I3DHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='3D',
                         **kwargs)


@HEADS.register_module()
class SlowFastHead(I3DHead):
    pass


@HEADS.register_module()
class GCNHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)
        
@HEADS.register_module()
class GCNHeadMV(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCNMV',
                         **kwargs)

@HEADS.register_module()
class GCNHeadMVAttn(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='ATTNMV',
                         **kwargs)

@HEADS.register_module()
class TSNHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='2D',
                         **kwargs)

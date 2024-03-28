import torch.nn as nn
import torch
import math
from mmcv.cnn import normal_init
from ..builder import HEADS
from .base import BaseHead

# network weights initialize
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / math.sqrt(in_dim / 2.)
    return torch.normal(size=size, std=xavier_stddev, mean=0.0)

class CVDM(nn.Module):
    def __init__(self,  num_classes):
        super(CVDM, self).__init__()

        self.num_classes = num_classes

        hcg_dim = 100 # graph classifier Cg() hidden layer dimension

        self.activation = nn.LeakyReLU(0.25)
        self.softmax = nn.Softmax()

        self.Classifier_g_W1 = nn.Parameter(xavier_init((num_classes * num_classes, hcg_dim)))
        self.Classifier_g_b1 = nn.Parameter(torch.zeros(hcg_dim))
        self.Classifier_g_W2 = nn.Parameter(xavier_init((hcg_dim, num_classes)))
        self.Classifier_g_b2 = nn.Parameter(torch.zeros(num_classes))

    def forward(self, p1, p2):
        p1 = p1.unsqueeze(-1)
        p2 = p2.unsqueeze(1)
        W_feature = torch.matmul(p1, p2)
        C_hw = W_feature.reshape((-1, self.num_classes * self.num_classes))
        C_h1 = self.activation(torch.matmul(C_hw, self.Classifier_g_W1) + self.Classifier_g_b1)
        classifier_g_logit = torch.matmul(C_h1, self.Classifier_g_W2) + self.Classifier_g_b2
        classifier_g_prob = self.softmax(classifier_g_logit)
        return classifier_g_logit, classifier_g_prob, W_feature


@HEADS.register_module()
class MVHeadCVDM(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='CVDM',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        
        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.in_c = in_channels

        self.n_views = 2
        self.n_persons = 2
        self.node_number = self.n_views * self.n_persons

        self.cvdm = CVDM(num_classes=num_classes)
        self.fc_cls = nn.Linear(self.in_c, num_classes)
        
    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        x = pool(x)

        x = x.reshape(N, M, C)

        # reshape by view
        x = x.reshape(N, self.n_views, self.n_persons, C)

        # mean on persons
        x = x.mean(dim=2)

        assert x.shape[2] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_scores = self.fc_cls(x)

        _, cls_score, _ = self.cvdm(cls_scores[:, 0, :], cls_scores[:, 1, :])

        return cls_score # TODO: pass W_feature matrix for norm in loss

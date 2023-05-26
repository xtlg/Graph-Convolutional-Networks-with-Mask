import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(support, adj)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_nodes = config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.mask = config['mask']
        self.dropout = nn.Dropout(0.2)
        self.gc11 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc12 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc13 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc14 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc15 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc21 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc22 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc23 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc24 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc25 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc31 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc32 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc33 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc34 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc35 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc41 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc42 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc43 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc44 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc45 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc51 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc52 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc53 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc54 = GraphConvolution(self.n_nodes, self.n_nodes)
        # self.gc55 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.fc1 = nn.Linear(self.in_features, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc4 = nn.Linear(self.n_hidden, self.out_features)

    def forward(self, x, pos, adj):
        # locate the missing position
        pos1 = pos[:, :self.n_nodes]
        pos2 = pos[:, self.n_nodes: 2 * self.n_nodes]
        # pos3 = pos[:, 2 * self.n_nodes: 3 * self.n_nodes]

        # split the input base the different features
        x1 = x[:, :self.n_nodes]
        x2 = x[:, self.n_nodes: 2 * self.n_nodes]

        # operation
        out1 = F.relu(self.gc11(x1, adj))
        out2 = F.relu(self.gc12(x2, adj))
        # using mask or not
        if self.mask:
            out1 = out1 * pos1 + x1
            out2 = out2 * pos2 + x2

        out1 = F.relu(self.gc21(out1, adj))
        out2 = F.relu(self.gc22(out2, adj))

        if self.mask:
            out1 = out1 * pos1 + x1
            out2 = out2 * pos2 + x2

        out1 = F.relu(self.gc31(out1, adj))
        out2 = F.relu(self.gc32(out2, adj))

        if self.mask:
            out1 = out1 * pos1 + x1
            out2 = out2 * pos2 + x2

        out1 = F.relu(self.gc41(out1, adj))
        out2 = F.relu(self.gc42(out2, adj))

        if self.mask:
            out1 = out1 * pos1 + x1
            out2 = out2 * pos2 + x2

        out1 = F.relu(self.gc51(out1, adj))
        out2 = F.relu(self.gc52(out2, adj))

        out = torch.cat((out1, out2), dim=1)

        if self.mask:
            out = out * pos + x

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.n_nodes = config['n_nodes']
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.fc3 = nn.Linear(self.in_features, self.in_features)
        self.fc4 = nn.Linear(self.in_features, self.in_features)
        self.fc5 = nn.Linear(self.in_features, self.in_features)
        self.fc6 = nn.Linear(self.in_features, self.n_hidden)
        self.fc7 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc8 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc9 = nn.Linear(self.n_hidden, self.out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)

        return x


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.n_nodes = config['n_nodes']
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.fc3 = nn.Linear(self.in_features, self.in_features)
        self.fc4 = nn.Linear(self.in_features, self.in_features)
        self.fc5 = nn.Linear(self.in_features, self.in_features)
        self.fc6 = nn.Linear(self.in_features, self.n_hidden)
        self.fc7 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc8 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc9 = nn.Linear(self.n_hidden, self.out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.fc1 = nn.Linear(self.in_features, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, 2 * self.n_hidden)
        self.fc3 = nn.Linear(2 * self.n_hidden, self.n_hidden)
        self.fc4 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc5 = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))

        return x


class MLP_M(nn.Module):
    def __init__(self, config):
        super(MLP_M, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.n_nodes = config['n_nodes']
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.fc3 = nn.Linear(self.in_features, self.in_features)
        self.fc4 = nn.Linear(self.in_features, self.in_features)
        self.fc5 = nn.Linear(self.in_features, self.in_features)
        self.fc6 = nn.Linear(self.in_features, self.n_hidden)
        self.fc7 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc8 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc9 = nn.Linear(self.n_hidden, self.out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, pos):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = out * pos + x
        out = F.relu(self.fc6(out))
        out = self.dropout(out)
        out = F.relu(self.fc7(out))
        out = F.relu(self.fc8(out))
        out = self.fc9(out)

        return out


class MLP_M_part_1(nn.Module):
    def __init__(self, config):
        super(MLP_M_part_1, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.n_nodes = config['n_nodes']
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.fc3 = nn.Linear(self.in_features, self.in_features)
        self.fc4 = nn.Linear(self.in_features, self.in_features)
        self.fc5 = nn.Linear(self.in_features, self.in_features)

    def forward(self, x, pos):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = out * pos + x
        return out


class MLP_M_part_2(nn.Module):
    def __init__(self, config):
        super(MLP_M_part_2, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.n_nodes = config['n_nodes']
        self.fc6 = nn.Linear(self.in_features, self.n_hidden)
        self.fc7 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc8 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc9 = nn.Linear(self.n_hidden, self.out_features)
        self.dropout = nn.Dropout(0.2)
        self.init_weights()

    def forward(self, x):
        out = F.relu(self.fc6(x))
        out = self.dropout(out)
        out = F.relu(self.fc7(out))
        out = F.relu(self.fc8(out))
        out = self.fc9(out)
        return out

    def init_weights(self):
        nn.init.constant_(self.fc6.weight, 1)
        nn.init.constant_(self.fc7.weight, 1)
        nn.init.constant_(self.fc8.weight, 1)
        nn.init.constant_(self.fc9.weight, 1)


class MLP_hy(nn.Module):
    def __init__(self, MLP_part1, MLP_part2):
        super(MLP_hy, self).__init__()
        self.net1 = MLP_part1
        for param in self.net1.parameters():
            param.requires_grad = False

        self.net2 = MLP_part2

    def forward(self, x, z):
        x = self.net1(x, z)
        x = self.net2(x)
        return x
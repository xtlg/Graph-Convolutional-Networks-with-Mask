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
        self.fc4 = nn.Linear(self.in_features, self.n_hidden)
        self.fc5 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc6 = nn.Linear(self.n_hidden, self.out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, pos, adj):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.relu(self.fc4(out))
        out = self.dropout(out)
        out = F.relu(self.fc5(out))
        out = self.fc6(out)
        out = out * pos + x

        return out


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.fc1 = nn.Linear(self.in_features, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc4 = nn.Linear(self.n_hidden, 1)

    def forward(self, x, pos, adj):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))

        return x


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.in_size = config['n_nodes'] * config['n_features']
        self.out_size = config['n_nodes'] * config['n_features']
        self.hidden = config['n_hidden']
        self.rnn = nn.RNN(self.in_size, self.hidden)
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.out_size)

    def forward(self, x, pos, adj):
        lstm_out, hidden_cell = self.rnn(x)
        out = F.relu(self.fc1(lstm_out[:, -1, :]))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.in_size = config['n_nodes'] * config['n_features']
        self.out_size = config['n_nodes'] * config['n_features']
        self.hidden = config['n_hidden']
        self.rnn = nn.LSTM(self.in_size, self.hidden)
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.out_size)

    def forward(self, x, pos, adj):
        lstm_out, hidden_cell = self.rnn(x)
        out = F.relu(self.fc1(lstm_out[:, -1, :]))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class GCNM(nn.Module):
    def __init__(self, config):
        super(GCNM, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_nodes = config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.dropout = nn.Dropout(0.2)
        self.gc11 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc12 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc21 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc22 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc31 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc32 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.fc1 = nn.Linear(self.in_features, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.out_features)

    def forward(self, x, pos, adj):
        # locate the missing position
        pos1 = pos[:, :self.n_nodes]
        pos2 = pos[:, self.n_nodes: 2 * self.n_nodes]

        # split the input base the different features
        x1 = x[:, :self.n_nodes]
        x2 = x[:, self.n_nodes: 2 * self.n_nodes]

        # operation
        out1 = F.relu(self.gc11(x1, adj))
        out2 = F.relu(self.gc12(x1, adj))
        out1 = out1 * pos1 + x1
        out2 = out2 * pos2 + x2
        out_1 = torch.cat((out1, out2), dim=1)

        out1 = F.relu(self.gc21(x1, adj))
        out2 = F.relu(self.gc22(x1, adj))
        out1 = out1 * pos1 + x1
        out2 = out2 * pos2 + x2
        out_2 = torch.cat((out1, out2), dim=1)

        out1 = F.relu(self.gc31(x1, adj))
        out2 = F.relu(self.gc32(x1, adj))
        out1 = out1 * pos1 + x1
        out2 = out2 * pos2 + x2
        out_3 = torch.cat((out1, out2), dim=1)

        out = F.relu(self.fc1(out_2))
        out = self.dropout(out)
        out = self.fc2(out)

        return out, out_1, out_2, out_3


class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_nodes = config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.dropout = nn.Dropout(0.2)
        self.gc11 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc12 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc21 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc22 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc31 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc32 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.fc1 = nn.Linear(self.in_features, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.out_features)

    def forward(self, x, pos, adj):
        # locate the missing position
        pos1 = pos[:, :self.n_nodes]
        pos2 = pos[:, self.n_nodes: 2 * self.n_nodes]

        # split the input base the different features
        x1 = x[:, :self.n_nodes]
        x2 = x[:, self.n_nodes: 2 * self.n_nodes]

        # operation
        out1 = F.relu(self.gc11(x1, adj))
        out2 = F.relu(self.gc12(x1, adj))
        out_1 = torch.cat((out1, out2), dim=1)

        out1 = F.relu(self.gc21(x1, adj))
        out2 = F.relu(self.gc22(x1, adj))
        out_2 = torch.cat((out1, out2), dim=1)

        out1 = F.relu(self.gc31(x1, adj))
        out2 = F.relu(self.gc32(x1, adj))
        out_3 = torch.cat((out1, out2), dim=1)

        out = F.relu(self.fc1(out_2))
        out = self.dropout(out)
        out = self.fc2(out)

        return out, out_1, out_2, out_3


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_nodes = config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.fc3 = nn.Linear(self.in_features, self.in_features)
        self.fc4 = nn.Linear(self.in_features, self.n_hidden)
        self.fc5 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc6 = nn.Linear(self.n_hidden, self.out_features)

    def forward(self, x, pos, adj):
        # operation
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(x))
        out3 = F.relu(self.fc3(x))
        out = F.relu(self.fc4(out3))
        out = self.dropout(out)
        out = F.relu(self.fc5(out))
        out = self.fc6(out)

        return out, out1, out2, out3


class MLPM(nn.Module):
    def __init__(self, config):
        super(MLPM, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_nodes = config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.fc3 = nn.Linear(self.in_features, self.in_features)
        self.fc4 = nn.Linear(self.in_features, self.n_hidden)
        self.fc5 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc6 = nn.Linear(self.n_hidden, self.out_features)

    def forward(self, x, pos, adj):
        # operation
        out1 = F.relu(self.fc1(x))
        out = out1 * pos + x
        out2 = F.relu(self.fc2(out))
        out = out2 * pos + x
        out3 = F.relu(self.fc3(out))
        out = out3 * pos + x
        out = F.relu(self.fc4(out))
        out = self.dropout(out)
        out = F.relu(self.fc5(out))
        out = self.fc6(out)

        return out, out1, out2, out3


class GCNR(nn.Module):
    def __init__(self, config):
        super(GCNR, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_nodes = config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.dropout = nn.Dropout(0.2)
        self.gc11 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc12 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc21 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc22 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc31 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.gc32 = GraphConvolution(self.n_nodes, self.n_nodes)
        self.fc1 = nn.Linear(self.in_features, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.out_features)

    def forward(self, x, pos, adj):
        # locate the missing position
        pos1 = pos[:, :self.n_nodes]
        pos2 = pos[:, self.n_nodes: 2 * self.n_nodes]

        # split the input base the different features
        x1 = x[:, :self.n_nodes]
        x2 = x[:, self.n_nodes: 2 * self.n_nodes]

        # operation
        out1 = F.relu(self.gc11(x1, adj))
        out2 = F.relu(self.gc12(x1, adj))
        out1 = out1 + x1
        out2 = out2 + x2
        out_1 = torch.cat((out1, out2), dim=1)

        out1 = F.relu(self.gc21(x1, adj))
        out2 = F.relu(self.gc22(x1, adj))
        out1 = out1 + x1
        out2 = out2 + x2
        out_2 = torch.cat((out1, out2), dim=1)

        out1 = F.relu(self.gc31(x1, adj))
        out2 = F.relu(self.gc32(x1, adj))
        out1 = out1 + x1
        out2 = out2 + x2
        out_3 = torch.cat((out1, out2), dim=1)

        out = F.relu(self.fc1(out_2))
        out = self.dropout(out)
        out = self.fc2(out)

        return out, out_1, out_2, out_3


class MLPR(nn.Module):
    def __init__(self, config):
        super(MLPR, self).__init__()
        self.in_features = config['n_features'] * config['n_nodes']
        self.out_features = config['n_features'] * config['n_nodes']
        self.n_nodes = config['n_nodes']
        self.n_hidden = config['n_hidden']
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.fc2 = nn.Linear(self.in_features, self.in_features)
        self.fc3 = nn.Linear(self.in_features, self.in_features)
        self.fc4 = nn.Linear(self.in_features, self.n_hidden)
        self.fc5 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc6 = nn.Linear(self.n_hidden, self.out_features)

    def forward(self, x, pos, adj):
        # operation
        out1 = F.relu(self.fc1(x))
        out = out1 + x
        out2 = F.relu(self.fc2(out))
        out = out2 + x
        out3 = F.relu(self.fc3(out))
        out = out3 + x
        out = F.relu(self.fc4(out))
        out = self.dropout(out)
        out = F.relu(self.fc5(out))
        out = self.fc6(out)

        return out, out1, out2, out3
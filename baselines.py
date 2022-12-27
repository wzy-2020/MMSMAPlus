import math
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


# MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, num_classes, bias=True), nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x

#DeepGOCNN
class CNN1D_DeepGoPlus(nn.Module):
    def __init__(self, input_dim=21, num_filters=16 * [256], filter_sizes=list(range(8, 129, 8)), num_classes=256):
        super(CNN1D_DeepGoPlus, self).__init__()
        # Define 1D convolutional layers
        cnn_layers = [
            nn.Conv1d(input_dim, num_filters[i], kernel_size=filter_sizes[i], padding=int(filter_sizes[i] / 2) - 1)
            for i in range(len(num_filters))]
        self.cnn = nn.ModuleList(cnn_layers)

        # Define global max pooling
        pool_layers = [nn.AdaptiveMaxPool1d(1) for _ in num_filters]
        self.globalpool = nn.ModuleList(pool_layers)

        # Define fully-connected layers
        self.fc_out = nn.Linear(sum(num_filters), num_classes)

    def forward(self, data):
        # Compute 1D convolutional part and apply global max pooling
        x = data.x
        all_x = []
        for cnn_layer, pool_layer in zip(self.cnn, self.globalpool):
            # all_x.append(pool_layer(cnn_layer(x)))
            all_x.append(pool_layer(cnn_layer(x)))

        # Concatenate all channels and flatten vector
        x = torch.cat(all_x, dim=1)
        x = torch.flatten(x, 1)
        # Compute fully-connected part
        output = self.fc_out(x)  # sigmoid in loss function
        output = torch.sigmoid(output)

        return output

# class ConvNet(nn.Module):
#     def __init__(self, DEVICE,GOXfile,transMatFile,input_dim, num_filters=16 * [256], filter_sizes=list(range(8, 129, 8)),num_classes = 453):
#         super().__init__()
#
#         cnn_layers = [
#             nn.Conv1d(input_dim, num_filters[i], kernel_size=filter_sizes[i], padding=int(filter_sizes[i] / 2) - 1)
#             for i in range(len(num_filters))]
#         self.cnn = nn.ModuleList(cnn_layers)
#
#         # Define global max pooling
#         pool_layers = [nn.AdaptiveMaxPool1d(1) for _ in num_filters]
#         self.globalpool = nn.ModuleList(pool_layers)
#         self.fc1 = nn.Linear(8192, 1024)
#         self.dro = nn.Dropout(0.5)
#         self.BN = nn.BatchNorm1d(8192, momentum=0.5)
#
#         #
#         self.in_channel = num_classes  # the number of node
#         self.gc1 = GraphConvolution(self.in_channel, 512)
#         self.gc2 = GraphConvolution(512, 1024)
#         self.gc3 = GraphConvolution(1024, 1024)
#         self.dataFile = transMatFile
#         self.filepro = scio.loadmat(self.dataFile)
#
#         self.transmat = self.filepro['A']
#         self.transmat = np.array(self.transmat)
#         self.transmat = torch.from_numpy(self.transmat).float()
#         self.transmat = self.transmat.to(DEVICE)
#         self.GOX = np.load(GOXfile, allow_pickle=True)
#         self.GOX = torch.from_numpy(self.GOX).float()
#         self.GOX = self.GOX.to(DEVICE)
#
#         self.relu = nn.LeakyReLU(0.2)
#
#         self.ff = nn.Linear(num_classes, num_classes)
#
#     def forward(self,data):
#         x = data.x
#         all_x = []
#         for cnn_layer, pool_layer in zip(self.cnn, self.globalpool):
#             all_x.append(pool_layer(cnn_layer(x)))
#         # Concatenate all channels and flatten vector
#         x = torch.cat(all_x, dim=1)
#         x = torch.flatten(x, 1)
#         out = self.BN(x)
#         out = self.fc1(out)
#         out = self.dro(out)
#         out = self.relu(out)
#         # link
#         gcout = self.gc1(self.GOX, self.transmat)
#         gcout = self.relu(gcout)
#         gcout = self.gc2(gcout, self.transmat)
#         gcout = self.relu(gcout)
#         gcout = self.gc3(gcout, self.transmat)
#         # mul
#         gcout = gcout.transpose(0, 1)
#         x = torch.matmul(out, gcout)
#         x = self.ff(x)
#         x = torch.sigmoid(x)
#         return x
#
# # DeepGOA
# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features, bias=False):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(1, 1, out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         support = torch.matmul(input, self.weight)
#         output = torch.matmul(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'

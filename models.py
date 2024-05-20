import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Orthogonal_Model(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers, mlp_in, mlp_hidden, num_class, num_domain, dropout):
        super(Orthogonal_Model, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.invariant_axis = nn.Parameter(torch.empty(mlp_in, mlp_in), requires_grad=True)
        self.relevant_axis = nn.Parameter(torch.empty(mlp_in, mlp_in), requires_grad=True)
        self.encoder = My_Encoder(in_channel, out_channel)
        self.toclass = My_MLP(n_layers, mlp_in * mlp_in, mlp_hidden, num_class, dropout)
        self.todomain = My_MLP(n_layers, mlp_in * mlp_in, mlp_hidden, num_domain, dropout)
        nn.init.xavier_uniform_(self.invariant_axis, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relevant_axis, gain=nn.init.calculate_gain('relu'))

    def similarity(self, feature1, feature2, type='cos'):
        sim = 0
        if type == 'cos':
            norm1 = torch.norm(feature1, dim=1, keepdim=True)
            norm2 = torch.norm(feature2, dim=1, keepdim=True)
            sim = feature1 * feature2 / (norm1 * norm2)
            sim = torch.sum(sim, dim=1)
            # sim = torch.mean(sim)
            sim = torch.max(torch.abs(sim))
        if type == 'pearson':
            sim = []
            for i in range(feature1.shape[0]):
                sim_temp = np.corrcoef(feature1[i].cpu().detach().numpy(), feature2[i].cpu().detach().numpy())
                # print(sim_temp)
                sim.append(sim_temp[0][1])
            sim = max(sim, key=abs)
        return sim

    def orthogonal_loss(self):
        invariant_axis = self.invariant_axis / torch.linalg.norm(self.invariant_axis, dim=1, keepdim=True)
        relevant_axis = self.relevant_axis / torch.linalg.norm(self.relevant_axis, dim=1, keepdim=True)
        o_loss = torch.mm(invariant_axis, relevant_axis.transpose(0, 1))
        o_loss = torch.sum(torch.abs(o_loss))
        return o_loss

    def forward(self, features):
        ## Encoder
        features = self.encoder(features)
        features = torch.reshape(features, (features.shape[0], -1))
        ## Orthogonal Decomposition
        if self.training:
            unit_invariant_axis = self.invariant_axis / (torch.linalg.norm(self.invariant_axis, dim=1, keepdim=True) ** 2)
            unit_relevant_axis = self.relevant_axis / (torch.linalg.norm(self.relevant_axis, dim=1, keepdim=True) ** 2)
            unit_invariant_axis = unit_invariant_axis.repeat(features.shape[0], 1)
            unit_relevant_axis = unit_relevant_axis.repeat(features.shape[0], 1)
            invariant_dot = torch.matmul(features, self.invariant_axis.transpose(0, 1)).reshape(-1, 1)
            relevant_dot = torch.matmul(features, self.relevant_axis.transpose(0, 1)).reshape(-1, 1)
            invariant_features = (invariant_dot * unit_invariant_axis).reshape(features.shape[0], -1)
            relevant_features = (relevant_dot * unit_relevant_axis).reshape(features.shape[0], -1)
            # print('Similarity: {:f}'.format(float(self.similarity(invariant_features, relevant_features, 'cos'))))
            invariant_class = self.toclass(invariant_features)
            invariant_domain = self.todomain(invariant_features)
            relevant_class = self.toclass(relevant_features)
            relevant_domain = self.todomain(relevant_features)

            return invariant_class, invariant_domain, relevant_class, relevant_domain
        else:
            unit_invariant_axis = self.invariant_axis / (torch.linalg.norm(self.invariant_axis, dim=1, keepdim=True) ** 2)
            unit_invariant_axis = unit_invariant_axis.repeat(features.shape[0], 1)
            invariant_dot = torch.matmul(features, self.invariant_axis.transpose(0, 1)).reshape(-1, 1)
            invariant_features = (invariant_dot * unit_invariant_axis).reshape(features.shape[0], -1)
            invariant_class = self.toclass(invariant_features)

            return invariant_class

class My_MLP(nn.Module):
    def __init__(self, n_layers, in_dim, hidden, out_dim, dropout):
        super(My_MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hidden))
        for _ in range(n_layers - 2):
            self.lins.append(nn.Linear(hidden, hidden))
        self.lins.append(nn.Linear(hidden, out_dim))
        self.dropout = dropout

        for i in range(len(self.lins)):
            nn.init.xavier_uniform_(self.lins[i].weight, gain=nn.init.calculate_gain('relu'))
    def forward(self, features):
        for i, lin in enumerate(self.lins[:-1]):
            features = lin(features)
            features = F.relu(features)
            features = F.dropout(features, p=self.dropout, training=self.training)
        features = self.lins[-1](features)
        return features

class My_Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(My_Encoder, self).__init__()
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels=in_channel, out_channels=1024, kernel_size=(1, 5)))
        self.pools.append(nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=False, ceil_mode=True))
        self.convs.append(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)))
        self.pools.append(nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=False, ceil_mode=True))
        self.convs.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)))
        self.pools.append(nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=False, ceil_mode=True))
        self.convs.append(nn.Conv2d(in_channels=256, out_channels=out_channel, kernel_size=(1, 1)))
        self.pools.append(nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=False, ceil_mode=True))

        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, features):
        features = features.unsqueeze(dim=2)
        for conv, pool in zip(self.convs, self.pools):
            features = conv(features)
            features = F.relu(features)
            features = pool(features)
        return features

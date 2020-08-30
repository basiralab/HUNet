import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from utils.topk_pool import TopKPooling


class HyperUNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.activation = kwargs['activation']
        self.depth = kwargs['depth']
        self.pool_ratios = kwargs['pool_ratios']

        # Only used when stacking HUNets
        self.should_drop = kwargs['should_drop']
        self.dropout = nn.Dropout(p=kwargs['dropout_rate'])

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()

        self.hunet_act = F.relu
        self.sum_res = kwargs['sum_res']
        self.H = np.array(kwargs['H_for_hunet'], dtype=np.float32)

        for i in range(self.depth):
            self.pools.append(TopKPooling(self.dim_in, self.pool_ratios))
            self.down_convs.append(HGNN_conv(self.dim_in, self.dim_in))

        for i in range(self.depth - 1):
            self.up_convs.append(HGNN_conv(self.dim_in, self.dim_in))
        self.up_convs.append(HGNN_conv(self.dim_in, self.dim_out))

    def forward(self, feat):
        x = feat
        xsaved = [x]
        graphs = [torch.Tensor(self.H).to('cuda:0')]
        perms = [range(len(x))]
        for i in range(1, self.depth + 1):
            x, batch, perm, _ = self.pools[i - 1](x, None)

            H = torch.tensor(np.array([self.H[perm[i], perm.cpu().numpy()] for i in range(len(perm))]).reshape(
                (len(perm), len(perm)))).to('cuda:0')
            x = self.down_convs[i - 1](x, H)
            x = self.hunet_act(x)

            if i < self.depth:
                xsaved += [x]
                graphs += [H]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - i - 1

            res = xsaved[j]
            H = graphs[j]
            perm = perms[j + 1]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.stack((res, up), dim=1)
            x = self.up_convs[i](x, H)
            x = self.activation(x) if i == self.depth - 1 else self.hunet_act(x)
        if self.should_drop:
            x = self.dropout(x)
        return x


class HGNN_conv(nn.Module):

    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, H: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = H.matmul(x)
        return x

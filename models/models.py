from models.layers import *


class HUNET(nn.Module):
    def __init__(self,**kwargs):

        super().__init__()
        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_stack = kwargs['n_stack']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        self.out_act = nn.LogSoftmax(dim=-1)
        self.dim_reduce = HGNN_conv(self.dim_feat,layer_spec[0])
        self.H = torch.Tensor(kwargs['H_for_hunet']).to(device='cuda:0')
        self.hunets = nn.ModuleList([HyperUNet(
            dim_in=layer_spec[i],
            dim_out=layer_spec[i+1],
            dropout_rate=kwargs['dropout_rate'],
            activation=nn.ReLU(),
            depth=kwargs["hunet_depth"],
            pool_ratios = kwargs['pool_ratios'],
            sum_res=True,
            should_drop=True,
            H_for_hunet=kwargs['H_for_hunet']) if i < kwargs['n_stack'] - 1 else HyperUNet(
            dim_in=layer_spec[i],
            dim_out=self.n_categories,
            dropout_rate=kwargs['dropout_rate'],
            activation=self.out_act,
            depth=kwargs["hunet_depth"],
            pool_ratios = kwargs['pool_ratios'],
            sum_res=True,
            should_drop=False,
            H_for_hunet=kwargs['H_for_hunet'])
                                 for i in range(kwargs['n_stack'])])

    def forward(self, **kwargs):
        x = kwargs['feats']
        x = self.dim_reduce(x,self.H)
        for i in range(len(self.hunets)):
            x = self.hunets[i](x)
        return x

import torch
import torch.nn as nn
from torchvision import models

# %% 
class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config

# %%    
class TSM(nn.Module):
    def __init__(self, n_segment=180, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)

# %%
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=32, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        nt, c, h, w = x.size()

        x = x.view(nt, c, h, w)

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 
        x = torch.cat([x0, x1, x2, x3], -1)  # B 4*C H/2 W/2 
        x = x.view(nt, -1, 4*c)  # B  H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(nt, 2*c, int(h/2),int(w/2))

        return x

#%%
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ratio = rotio

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

#%%
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    


#%%
class TA2N(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=180, img_size=72, channel='raw'):
        super(TA2N, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.img_size = img_size
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                  bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True,padding=(1, 1))
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, bias=True,padding=(1, 1))
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True,padding=(1, 1))
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 49:
            self.final_dense_1 = nn.Linear(34560, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(20736, self.nb_dense, bias=True)     #20736     
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(36864, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)     #defaut 2
        self.batch_norm = nn.BatchNorm2d(in_channels)       #default 3
        self.channel = channel
        self.channelAttention1 = ChannelAttention(self.nb_filters1)      #inplace = self.nb_filters1
        self.channelAttention2 = ChannelAttention(self.nb_filters2)
        self.spatialAttention = SpatialAttention()
        self.residual_cnn1 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True,padding=(1, 1))
        self.residual_cnn2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True,padding=(1, 1))
        self.residual_cnn3 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True,padding=(1, 1))
        self.residual_cnn4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True,padding=(1, 1))

    def forward(self, inputs, params=None):
        if self.channel == 'raw':
            inputs = torch.diff(inputs, dim=0)
            inputs = torch.cat([inputs,inputs[-1].reshape(1,self.in_channels,self.img_size,self.img_size)], dim=0)
            inputs = self.batch_norm(inputs)   
        
        #attention block1           
        inputs = self.TSM_1(inputs)
        inputs = self.motion_conv1(inputs)
        inputs = torch.tanh(inputs)
        
        short_cut = inputs
        inputs = self.residual_cnn1(inputs)
        inputs = self.channelAttention1(inputs)*inputs
        inputs = self.spatialAttention(inputs)*inputs
        inputs = inputs+short_cut

        #attention block2
        inputs = self.TSM_2(inputs)
        inputs = self.motion_conv2(inputs)
        inputs = torch.tanh(inputs)

        short_cut = inputs
        inputs = self.residual_cnn1(inputs)
        inputs = self.channelAttention1(inputs)*inputs
        inputs = self.spatialAttention(inputs)*inputs
        inputs = inputs+short_cut
        
        #apperance_attention
        g = torch.sigmoid(self.apperance_att_conv1(inputs))
        g = self.attn_mask_1(g)
        inputs = inputs * g

        inputs = self.avg_pooling_1(inputs)
        inputs = self.dropout_1(inputs)
        
        #attention block3
        inputs = self.TSM_3(inputs)
        inputs = self.motion_conv3(inputs)
        inputs = torch.tanh(inputs)
        
        short_cut = inputs
        inputs = self.residual_cnn3(inputs)
        inputs = self.channelAttention2(inputs)*inputs
        inputs = self.spatialAttention(inputs)*inputs
        inputs = inputs+short_cut

        #attention block4
        inputs = self.TSM_4(inputs)
        inputs = self.motion_conv4(inputs)
        inputs = torch.tanh(inputs)
        
        short_cut = inputs
        inputs = self.residual_cnn4(inputs)
        inputs = self.channelAttention2(inputs)*inputs
        inputs = self.spatialAttention(inputs)*inputs
        inputs = inputs+short_cut

        g = torch.sigmoid(self.apperance_att_conv2(inputs))
        g = self.attn_mask_2(g)
        inputs = inputs * g


        inputs = self.avg_pooling_3(inputs)
        inputs = self.dropout_3(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        inputs = torch.tanh(self.final_dense_1(inputs))
        inputs = self.dropout_4(inputs)
        inputs = self.final_dense_2(inputs)
        return inputs
    

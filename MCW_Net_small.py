
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import math
from torch.autograd import Function



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBlock(nn.Module):
    def __init__(self, channel,reduction = 16):
        super(SEBlock, self).__init__()
        # self.conv1  = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1    = nn.BatchNorm2d(channel)
        # self.relu   = nn.ReLU(inplace=True)
        
        self.prelu  = nn.PReLU()
        self.conv2  = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(channel)
        self.se     = SELayer(channel, reduction)
    def forward(self,x):
        residual = x
        out  = x 
        # out = self.conv1(out)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.prelu(out)

        return out


def kaiming_uniform_small(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only 
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    bound = bound / 100
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class _DCR_block(nn.Module):
    def __init__(self, channel_in, inter_channels = None):
        super(_DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in*2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)

        return out

class RegionNONLocalBlock(nn.Module):
    def __init__(self, in_channels, grid=[8, 8]):
        super(RegionNONLocalBlock, self).__init__()

        self.non_local_block = _NonLocalBlock2D(in_channels, sub_sample=True, bn_layer=False)
        self.grid = grid

    def forward(self, x):
        batch_size, _, height, width = x.size()

        input_row_list = x.chunk(self.grid[0], dim=2)

        # print(input_row_list[0].shape)

        output_row_list = []
        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = row.chunk(self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                grid = self.non_local_block(grid)
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)
        return output

class _NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=False):
        super(_NonLocalBlock2D, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0) # padding = 0

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0), # padding = 0
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0) # padding = 0
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0) # padding = 0
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0) # padding = 0

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, inter_channels=32, dilation=[1, 1, 1]):
        super(DenseBlock, self).__init__()
        
        num_convs = len(dilation)
        concat_channels = in_channels + num_convs * inter_channels
        self.conv_list, channels_now = nn.ModuleList(), in_channels
        
        for i in range(num_convs):
            conv = nn.Sequential(
                nn.Conv2d(in_channels=channels_now, out_channels=inter_channels, kernel_size=3,
                          stride=1, padding=dilation[i], dilation=dilation[i]),
                nn.ReLU(inplace=True),
            )
            self.conv_list.append(conv)
            
            channels_now += inter_channels

        assert channels_now == concat_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, x, local_skip=True):
        feature_list = [x,]

        for conv in self.conv_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = conv(inputs)
            feature_list.append(outputs)

        inputs = torch.cat(feature_list, dim=1)
        fusion_outputs = self.fusion(inputs)
        
        if local_skip:
            block_outputs = fusion_outputs + x
        else:
            block_outputs = fusion_outputs

        return block_outputs

class _SuperResolution(nn.Module):
    def __init__(self, base_ch=16, inter_ch=7777):
        super(_SuperResolution, self).__init__()

        self.conv1 = nn.Conv2d(3      , base_ch, 1, 1, 0)
        self.conv2 = nn.Conv2d(base_ch, base_ch, 3, 1, 1)
        self.relu2 = nn.PReLU()


        self.dense_1 = _DCR_block(channel_in=base_ch, inter_channels=None)
        self.dense_2 = _DCR_block(channel_in=base_ch, inter_channels=None)
        self.dense_3 = _DCR_block(channel_in=base_ch, inter_channels=None)
        self.dense_4 = _DCR_block(channel_in=base_ch, inter_channels=None)
       

        self.merge   = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch, 1, 1, 0),
            nn.Conv2d(base_ch    , base_ch, 3, 1, 1),
            nn.PReLU()
        )
        
        
        self.final   = nn.Sequential(
            nn.Conv2d(base_ch    , base_ch, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(base_ch    , base_ch, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(base_ch    , 3      , 1, 1, 0),
        )
        
        
        
          
        for m in self.modules():
           if isinstance(m, nn.Conv2d):
               kaiming_uniform_small(m.weight, mode='fan_out', nonlinearity='relu')
           elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)

        
        
    def forward(self, x):
    
        shallow_f = x 
        conv1     = self.conv1(x)
        feature_0 = self.relu2(self.conv2(conv1))
        
        dout_1 = self.dense_1(feature_0)
        dout_2 = self.dense_2(dout_1)
        dout_3 = self.dense_3(dout_2)
        dout_4 = self.dense_4(dout_3)
        
        out = torch.cat(
            (dout_1, dout_2, dout_3, dout_4), dim=1)

        out = self.merge(out)
        out += conv1
        
        output = self.final(out)
        output += shallow_f
        
        return output


class _down(nn.Module):
    def __init__(self, channel_in, inter_channels = None):
        super(_down, self).__init__()

        self.relu = nn.PReLU()
        #self.maxpool = nn.MaxPool2d(2)\
        self.dwt      = DWT()
        self.conv = nn.Conv2d(in_channels=4*channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.dwt(x)

        out = self.relu(self.conv(out))

        return out


class _up(nn.Module):
    def __init__(self, channel_in, inter_channels = None):
        super(_up, self).__init__()

        self.relu = nn.PReLU()
        #self.subpixel = nn.PixelShuffle(2)
        self.iwt       = IWT()
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
    
        out = self.relu(self.conv(x))
        out = self.iwt(out)

        return out




class  _concat_SE_conv(nn.Module):
    def __init__(self, channel_in, channel_out, reduction=4):
        super( _concat_SE_conv , self).__init__()
        self.relu     = nn.PReLU()
        self.se_layer = SELayer(channel_in, reduction)
        self.conv     = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=1, padding=0)
    def forward(self, x ):
        residual = x
        out = self.se_layer(x) + residual
        out = self.relu( self.conv(out))
        return out
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.name = os.path.basename(__file__ ).split(".")[0]
        #############################################Encoder#############################################
        oc = 16
        reduction = 4
        
        # Level 1
        self.conv_i = nn.Conv2d(in_channels=3, out_channels=oc, kernel_size=1, stride=1, padding=0)
        self.relu1  = nn.PReLU()
        self.DCR_block11 = self.make_layer(_DCR_block, oc)
        self.DCR_block12 = self.make_layer(_DCR_block, oc)
        self.NonLocalBlock2D12  = RegionNONLocalBlock(oc , grid=[16, 4])
        self.down1    = self.make_layer(_down, oc)
        self.tran1to2 = self.make_layer(_down, oc)
        self.tran1to3 = self.make_layer([_down,_down], [oc , oc*2])
        self.tran1to4 = self.make_layer([_down,_down,_down], [oc , oc*2, oc*4] , 4)


        # Level 2 
        self.DCR_block21 = self.make_layer(_DCR_block, oc*2)
        self.DCR_block22 = self.make_layer(_DCR_block, oc*2)
        self.NonLocalBlock2D22  = RegionNONLocalBlock(oc*2 , grid=[8, 2])
        self.down2    = self.make_layer(_down         , oc*2)
        self.tran2to1 = self.make_layer(_up           , oc*2)
        self.tran2to3 = self.make_layer(_down         , oc*2)
        self.tran2to4 = self.make_layer([_down,_down] , [oc*2, oc*4])

        # Level 3
        self.DCR_block31 = self.make_layer(_DCR_block, oc*4)
        self.DCR_block32 = self.make_layer(_DCR_block, oc*4)
        self.NonLocalBlock2D32  = RegionNONLocalBlock(oc*4 , grid=[4, 1])
        self.down3    = self.make_layer(_down    , oc*4)
        self.tran3to1 = self.make_layer([_up,_up],[oc*4 , oc*2])
        self.tran3to2 = self.make_layer(_up      , oc*4)
        self.tran3to4 = self.make_layer(_down    , oc*4)
        
        # Level 4 
        self.se_connect4        = _concat_SE_conv( oc*32 , oc*8 )
        self.DCR_block41        = self.make_layer(_DCR_block, oc*8)
        self.DCR_block42        = self.make_layer(_DCR_block, oc*8)
        self.NonLocalBlock2D42  = RegionNONLocalBlock(oc*8 , grid=[4, 1])
        self.up3                = self.make_layer(_up          , oc*8)
        # self.finaltran4to1      = self.make_layer([_up,_up,_up], [oc*8 , oc*4, oc*2])

        #############################################Decoder#############################################
        #self.NonLocalBlock2D32 = self.make_layer(RegionNONLocalBlock, oc*4)
        self.se_connect3        = _concat_SE_conv( oc*16 , oc*4 )
        self.DCR_block33 = self.make_layer(_DCR_block, oc*4)
        self.DCR_block34 = self.make_layer(_DCR_block, oc*4)
        self.NonLocalBlock2D34 = RegionNONLocalBlock(oc*4 , grid=[4, 1])     




        
        self.up2               = self.make_layer(_up, oc*4)    
        self.se_connect2       = _concat_SE_conv( oc*8 , oc*2 )
        self.DCR_block23 = self.make_layer(_DCR_block, oc*2)
        self.DCR_block24 = self.make_layer(_DCR_block, oc*2)
        self.NonLocalBlock2D24 = RegionNONLocalBlock(oc*2 , grid=[8, 2])
        
        self.se_connect1        = _concat_SE_conv( oc*4 , oc*1 )
        self.up1                = self.make_layer(_up, oc*2)
        self.DCR_block13 = self.make_layer(_DCR_block, oc)
        self.DCR_block14 = self.make_layer(_DCR_block, oc)
        self.NonLocalBlock2D15 = RegionNONLocalBlock(oc*1 , grid=[16, 4])
        
        self.DCR_block16 = self.make_layer(_DCR_block, oc)
        self.conv_f = nn.Conv2d(in_channels=oc, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()
        #self.SuperResolution = self.make_layer(_SuperResolution, 16, 16)

    def make_layer(self, block, channel_in, inter_channels = None):
        layers = []
        if isinstance(block, list):
            for i in range(len(block)):
                b = block[i]
                c = channel_in[i]
                layers.append(b(c))
        elif True:#isinstance(block, nn.Module):
            layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        #############################################Encoder#############################################

        ################           Level1  #######################
        out      = self.relu1(self.conv_i(x))
        out      = self.DCR_block11(      out)
        out      = self.DCR_block12(      out)
        out1     = self.NonLocalBlock2D12(out)                 # oc
        out      = self.down1(           out1)
        tran1to2 = self.tran1to2(        out1)
        tran1to3 = self.tran1to3(        out1)
        tran1to4 = self.tran1to4(        out1)
        ################           Level2  #######################

        out      = self.DCR_block21(      out)
        out      = self.DCR_block22(      out)
        out2     = self.NonLocalBlock2D22(out)                 #2oc
        out      = self.down2(           out2)
        tran2to1 = self.tran2to1(        out2)
        tran2to3 = self.tran2to3(        out2)
        tran2to4 = self.tran2to4(        out2)

        ################           Level3  #######################

        out      = self.DCR_block31(      out)
        out      = self.DCR_block32(      out)
        out3     = self.NonLocalBlock2D32(out)                   #4oc
        out      = self.down3(           out3)                       #8oc
        tran3to1 = self.tran3to1(        out3)
        tran3to2 = self.tran3to2(        out3)
        tran3to4 = self.tran3to4(        out3)
        
        
        #############################################Decoder#############################################

        ################           Level4  #######################
        mlc4   = torch.cat( [tran1to4  ,tran2to4 , tran3to4 ,out ] , 1 )   #multi-level connection
        mlc4   = self.se_connect4( mlc4)
        conc4  = mlc4 
        out    = self.DCR_block41(conc4)
        out    = self.DCR_block42(out)
        out4   = self.NonLocalBlock2D42(out)
        out    = out4 + conc4
   
        
        ################           Level3  #######################
        out    = self.up3(out)  
        mlc3   = torch.cat( [tran1to3  ,tran2to3 , out3 , out  ] , 1 )   
        mlc3   = self.se_connect3( mlc3)
        out    = mlc3
        out    = self.DCR_block33(out)
        out    = self.DCR_block34(out)
        out    = self.NonLocalBlock2D34(out)                      #4oc

        
        ################           Level2  #######################
        out    = self.up2(out)                    #2oc
        mlc2   = torch.cat( [tran1to2 ,out2 , tran3to2 ,out  ] , 1 )   
        mlc2   = self.se_connect2( mlc2)
        out    = mlc2 
        out    = self.DCR_block23(out)
        out    = self.DCR_block24(out)
        out    = self.NonLocalBlock2D24(out)
        
        ################           Level1  #######################    
        out    = self.up1(out)
        mlc1   = torch.cat( [out1 ,tran2to1 , tran3to1 , out ] , 1 )   
        mlc1   = self.se_connect1( mlc1)
        out    = mlc1
        out    = self.DCR_block13(out)
        out    = self.DCR_block14(out)
        out    = self.NonLocalBlock2D15(out)


        out = self.DCR_block16(out)
        attention = self.relu2(self.conv_f(out))
        out = torch.add(residual, attention)


        return attention, out



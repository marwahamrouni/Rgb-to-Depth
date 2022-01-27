import torch
from torch import nn
import torch.nn.functional as F

class ContractingBlock(nn.Module):

    def __init__(self, input_channels, kernel_size=3, stride=1, padding=1, use_dropout = False, use_bn = True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3, stride=1, padding = 1,bias=True, padding_mode = 'reflect')
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3, stride=1, padding = 1, padding_mode = 'reflect')
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = nn.MaxPool2d(2, 2)
        if use_dropout :
            self.dropout = nn.Dropout(0.2)
        self.use_dropout = use_dropout
        if use_bn:
            self.batchnorm1 = nn.BatchNorm2d(input_channels*2)
            self.batchnorm2 = nn.BatchNorm2d(input_channels*2)
        self.use_bn = use_bn
    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm1(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.downsample(x)
        
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, output_channels,use_bn=True, use_dropout=False):
        super(ExpandingBlock, self).__init__()
        self.conv2 = nn.Conv2d(input_channels, output_channels, 3, padding=1, padding_mode = 'reflect')
        self.conv3 = nn.Conv2d(output_channels, output_channels, 3, padding=1, padding_mode = 'reflect')
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        if use_dropout :
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
        if use_bn:
            self.batchnorm1 = nn.BatchNorm2d(output_channels)
            self.batchnorm2 = nn.BatchNorm2d(output_channels)
        self.use_bn = use_bn
    def forward(self, x, skip_con_x):
        x =  F.interpolate(x, size=[skip_con_x.size(2), skip_con_x.size(3)], mode='bilinear', align_corners=True)
        x=torch.cat([x, skip_con_x], axis=1)
        x=self.conv2(x)
        if self.use_bn:
            x=self.batchnorm1(x)
        if self.use_dropout:
            x=self.dropout(x)
        x=self.activation(x)
        x=self.conv3(x)
        if self.use_bn:
            x=self.batchnorm2(x)
        if self.use_dropout:
            x=self.dropout(x)
        x=self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = self.conv(x)
        return x

##Residual block##
class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False, padding_mode = 'reflect')
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.leakyelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False, padding_mode = 'reflect')
        self.bn1_2 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False, padding_mode = 'reflect')
        self.bn2 = nn.BatchNorm2d(num_output_features)
    def forward(self, x, skip_con_x):
        x = F.interpolate(x, size=[skip_con_x.size(2), skip_con_x.size(3)], mode='nearest')
        x=torch.cat([x, skip_con_x], axis=1)
        x_conv1 = self.leakyelu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.leakyelu(bran1 + bran2)

        return out

## The generator network##
class UNet(nn.Module):
    def __init__(self, out_channels, encoder, bottleneck_channels=512):
        super(UNet, self).__init__()
        self.encoder = encoder
        self.expand1 = _UpProjection(bottleneck_channels // 1 + 112 + 64, bottleneck_channels//2)# All the '_UpProjection 'blocks should be replaced by 'ExpandingBlock'
        self.expand2 = _UpProjection(bottleneck_channels // 2 + 40 + 24, bottleneck_channels//4) # when the network is trained on the KITTI dataset
        self.expand3 = _UpProjection(bottleneck_channels // 4 + 24 + 16, bottleneck_channels//8)
        self.expand4 = _UpProjection(bottleneck_channels // 8 + 16 + 8, bottleneck_channels//16)
        self.expand5 = _UpProjection(bottleneck_channels//16 +3, bottleneck_channels//32)
        self.downFeature = FeatureMapBlock(bottleneck_channels//32, out_channels)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        features = self.encoder(x)
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[1], features[2], features[3], features[4]
        x7 = self.expand1(x_block4, x_block3)
        x8 = self.expand2(x7, x_block2)
        x9 = self.expand3(x8, x_block1)
        x10 = self.expand4(x9, x_block0)
        x11 = self.expand5(x10, x)
        xf = self.downFeature(x11)
        return self.sigmoid(xf) # Replace self.sigmoid(xf) by 0.3 * self.sigmoid(xf) in the case of unsupervised training
                                # to enforce the estimated disparity map to be a maximum of 0.3

## The critic network##
class Critic(nn.Module):
    def __init__(self, input_channels, hidden_channels=8):
        super(Critic, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = FeatureMapBlock(hidden_channels*16, 1)


    def forward(self, x, y):
        x = torch.cat([x, y], axis = 1) # this line must be removed in the case of training the unsupervised method as the critic evaluates the reconstructed image alone.
        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)  
        x = self.contract4(x) 
        x = self.final(x)
        return(x.view(len(x),-1))


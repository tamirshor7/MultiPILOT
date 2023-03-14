import torch
from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F
from models.rec_models.ACNN.common.utils import fft2, ifft2
from data import transforms


class ConvBlockBn(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
               f'drop_prob={self.drop_prob})'


class AttentionWeightChannel(nn.Module):

    def __init__(self, w, h, channel_num):
        super(AttentionWeightChannel, self).__init__()
        self.w = int(w)
        self.h = int(h)
        self.c = channel_num
        self.r = 8

        self.pool = nn.AvgPool2d((self.w, self.h))
        self.fc1 = nn.Linear(channel_num, channel_num)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel_num // self.r, channel_num)
        self.sig = nn.Sigmoid()
        self.cs_weight = nn.Conv2d(channel_num, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, inputs):
        x = torch.abs(inputs)
        cs_x = self.cs_weight(inputs)
        cs_x = self.sig(cs_x)
        inputs_c = cs_x * inputs

        x = self.pool(x)
        x = self.fc1(x.view(-1, x.shape[1]))

        weight = self.sig(x).view(-1, x.shape[1], 1, 1)
        inputs_s = weight * inputs

        output = torch.max(inputs_c, inputs_s)
        return output, cs_x.detach(), weight.detach()


class AcnnModel(BasicModule):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, slice_num, isweight):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super(AcnnModel, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.isweight = isweight
        self.relu = torch.nn.ReLU().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.fft = fft2
        self.ifft = ifft2
        ConvBlock = ConvBlockBn

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        w = 384
        h = 144
        self.down_sample_layers += [AttentionWeightChannel(w, h, ch)]
        w /= 2
        h /= 2
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            self.down_sample_layers += [AttentionWeightChannel(w, h, ch * 2)]

            ch *= 2
            w /= 2
            h /= 2

        self.conv = ConvBlock(ch, ch, drop_prob)
        self.mid_attention = AttentionWeightChannel(w, h, ch)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            w *= 2
            h *= 2
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            self.up_sample_layers += [AttentionWeightChannel(w, h, ch // 2)]
            ch //= 2

        w *= 2
        h *= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.up_sample_layers += [AttentionWeightChannel(w, h, ch)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans // slice_num, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        c_weight_list = []
        c_weight_list2 = []
        output_ori = input

        output_ori = self.fft(output_ori)
        # output_ori = torch.cat((output_ori.real.unsqueeze(-1),output_ori.imag.unsqueeze(-1)),dim=-1)
        ks = output_ori
        output_ori = torch.cat((output_ori[:, :, :, :, 0], output_ori[:, :, :, :, 1]), 1)
        # output_ori = torch.sqrt(torch.sum(output_ori**2,dim=-1))
        output = output_ori

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            if layer.__class__.__name__ == 'ConvBlockBn':
                output = layer(output)
            else:
                output, c_weight, c_weight2 = layer(output)
                c_weight_list.append(c_weight)
                c_weight_list2.append(c_weight2)
                stack.append(output)
                output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)
        output, c_weight_mid, c_weight_2 = self.mid_attention(output)
        c_weight_list.append(c_weight_mid)
        c_weight_list2.append(c_weight_2)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            if layer.__class__.__name__ == 'ConvBlockBn':
                output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
                output = torch.cat([output, stack.pop()], dim=1)
                output = layer(output)
            else:
                output, c_weight_up, c_weight2 = layer(output)
                c_weight_list.append(c_weight_up)
                c_weight_list2.append(c_weight2)

        output = self.conv2(output)
        
        slice_num = 1#input.shape[1] // 8
        half_c = input.shape[1]
        mid_slice = slice_num // 2
        output = output + torch.cat([output_ori[:,half_c * mid_slice:half_c * (mid_slice + 1),:,:], output_ori[:,half_c + half_c * mid_slice:half_c + half_c * (mid_slice + 1),:,:]], 1)#torch.cat([output_ori[:,:8,:,:], output_ori[:,half_c:half_c + 8,:,:]], 1)



        channal_num = output.shape[1]
        output = torch.stack((output[:, :channal_num // 2, :, :], output[:, channal_num // 2:, :, :]), 4)
        ke = output
        output = self.ifft(output)
        #output = transforms.complex_center_crop(output, (384, 144))
        output = transforms.complex_abs(output)
        #output, _, _ = transforms.normalize_instance(output, eps=1e-11)
        # output = self.relu(output)

        return output, ks, ke, c_weight_list, c_weight_list2

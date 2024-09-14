import torch
import torch.nn as nn
from torch.nn import functional as F

def sqrt4(n):
    """
    input:
    n is 2^k

    output:
    if n is 4^k
        return sqrt(n)
    else 
        return sqrt(n/2)
    """
    if n & (n - 1) == 0 and (n & 0xAAAAAAAA) == 0:
        return int(n**0.5)
    else:
        return int((n / 2)**0.5)

class ChannelAttention(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(inplanes, inplanes // reduction, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inplanes // reduction, inplanes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, inplanes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplanes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out

class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn=None
        if norm:
            self.bn = nn.BatchNorm2d(outplanes)
        self.attention = CBAM(outplanes, reduction=sqrt4(outplanes))

    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)
        if self.bn is not None:
            fx = self.bn(fx)
        fx = self.attention(fx)
        return fx


class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)
        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
        self.attention = CBAM(outplanes, reduction=sqrt4(outplanes))

    def forward(self, x):
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)
        if self.dropout is not None:
            fx = self.dropout(fx)
        fx = self.attention(fx)
        return fx

class UnetGenerator(nn.Module):
    def __init__(self,):
        super().__init__()
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 1024)
        self.encoder6 = EncoderBlock(1024, 1024)
        self.encoder7 = EncoderBlock(1024, 1024)
        self.encoder8 = EncoderBlock(1024, 1024, norm=False)

        self.decoder8 = DecoderBlock(1024, 1024, dropout=True)
        self.decoder7 = DecoderBlock(2*1024, 1024, dropout=True)
        self.decoder6 = DecoderBlock(2*1024, 1024, dropout=True)
        self.decoder5 = DecoderBlock(2*1024, 512)
        self.decoder4 = DecoderBlock(2*512, 256)
        self.decoder3 = DecoderBlock(2*256, 128)
        self.decoder2 = DecoderBlock(2*128, 64)
        self.decoder1 = nn.ConvTranspose2d(2*64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)

        return torch.relu(d1)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.conv(x)
        if self.isn is not None:
            fx = self.isn(fx)
        fx = self.lrelu(fx)
        return fx

class Discriminator(nn.Module):
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(6, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.blockf = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.blockf(fx)
        return fx
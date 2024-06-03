import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as nF
########################################### ASPP #################################################
class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        return self.block(x)

class ASPPPolling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPolling, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.block(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1x1 = ASPPConv(in_channels, inner_channels, 1, 1, 0)
        self.conv3x3_rate6 = ASPPConv(in_channels, inner_channels, 3, 6, 6)
        self.conv3x3_rate12 = ASPPConv(in_channels, inner_channels, 3, 12, 12)
        self.conv3x3_rate18 = ASPPConv(in_channels, inner_channels, 3, 18, 18)
        self.img_pooling = ASPPPolling(in_channels, inner_channels)

        self.outputconv = nn.Sequential(
            nn.Conv2d(inner_channels * 5, out_channels, 1, bias = False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        conv1x1 = self.conv1x1(x)
        conv3x3_rate6 = self.conv3x3_rate6(x)
        conv3x3_rate12 = self.conv3x3_rate12(x)
        conv3x3_rate18 = self.conv3x3_rate18(x)
        img_pooling = nF.interpolate(self.img_pooling(x), size=size, mode="bilinear")

        return self.outputconv(
            torch.cat(
                (
                    conv1x1,
                    conv3x3_rate6, 
                    conv3x3_rate12, 
                    conv3x3_rate18, 
                    img_pooling
                ), 
                dim=1
            )
        )

########################################### ASPP #################################################

########################################## MyUnet ###############################################
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            mid_channel,
            out_channel,
            kernel_size,
            dilation,
            padding,
            upsample, 
            residual = False
        ):
        super(DecoderBlock, self).__init__()
        self.residual = residual
        self.upsample = upsample
        modules = [
            nn.Conv2d(
                in_channel,
                mid_channel,
                kernel_size = kernel_size,
                dilation = dilation,
                padding=padding
            ), 
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                mid_channel,
                mid_channel,
                kernel_size = kernel_size,
                dilation = dilation,
                padding=padding
            ), 
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                mid_channel,
                out_channel,
                kernel_size = kernel_size,
                dilation = dilation,
                padding=padding
            ), 
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        ]
        
        if self.upsample:
            self.upsample_unit = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )
        
        if self.residual:
            self.residual_conv = nn.Conv2d(in_channel, out_channel, 1)
        
        self.block = nn.Sequential(*modules)
        
    def forward(self, x):
        if self.upsample:
            if self.residual:
                return self.upsample_unit(self.block(x) + self.residual_conv(x))
            else:
                return self.upsample_unit(self.block(x))
        else:
            if self.residual:
                return self.block(x) + self.residual_conv(x)
            else:
                return self.block(x)

class Unet(nn.Module):
    def __init__(self, in_channel, out_classes):
        super(Unet, self).__init__()
        
        resnet = models.resnet34(weights = "DEFAULT")
        
        self.inconv = nn.Conv2d(in_channel, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace = True)
        
        # encoder
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # bridge
        self.bridge = ASPP(in_channels=512, inner_channels=256, out_channels=512)

        # decoder
        self.decoder4 = DecoderBlock(
            in_channel=1024,
            mid_channel=512,
            out_channel=256, 
            kernel_size=3, 
            dilation=1, 
            padding=1, 
            upsample=True,  
            residual=True
        )
        self.decoder3 = DecoderBlock(
            in_channel=512,
            mid_channel=256,
            out_channel=128,
            kernel_size=3,
            dilation=1,
            padding=1,
            upsample=True,
            residual=True
        )       
        self.decoder2 = DecoderBlock(
            in_channel=256,
            mid_channel=128,
            out_channel=64,
            kernel_size=3,
            dilation=1,
            padding=1,
            upsample=True, 
            residual=True
        )        
        self.decoder1 = DecoderBlock(
            in_channel=128,
            mid_channel=64,
            out_channel=64,
            kernel_size=3,
            dilation=1,
            padding=1,
            upsample=False,
            residual=True
        )

        self.final_conv = nn.Conv2d(64, out_classes, 3, padding=1)

        # side out
        self.side_out_b  = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear"), 
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1)
        )

        self.side_out_d4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear"), 
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        )

        self.side_out_d3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), 
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)
        )

        self.side_out_d2 = nn.Sequential(
            nn.Upsample(scale_factor=1, mode="bilinear"), 
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # ResUnet
        hx = self.inrelu(
            self.inbn(
                self.inconv(x)
            )
        )
        
        e1 = self.encoder1(hx)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bridge(e4)
        
        d4 = self.decoder4(torch.cat((e4, b), dim=1)) 
        d3 = self.decoder3(torch.cat((e3, d4), dim=1)) 
        d2 = self.decoder2(torch.cat((e2, d3), dim=1)) 
        d1 = self.decoder1(torch.cat((e1, d2), dim=1))

        # side out
        out_b  = self.side_out_b(b)
        out_d4 = self.side_out_d4(d4)
        out_d3 = self.side_out_d3(d3)
        out_d2 = self.side_out_d2(d2)

        out = self.final_conv(d1)
        
        return torch.sigmoid(out), torch.sigmoid(out_d2), torch.sigmoid(out_d3), torch.sigmoid(out_d4), torch.sigmoid(out_b) 

########################################## MyUnet ###############################################
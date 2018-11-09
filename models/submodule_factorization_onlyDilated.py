from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size, stride,downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, kernel_size, stride, pad, dilation),
                                   nn.ReLU(inplace=True))
        #print('inplanes:',inplanes)
        #print('out_planes:',planes)

        self.conv2 = convbn(planes, planes, kernel_size, 1, pad, dilation)

        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        #print('x:',x.shape)
        out = self.conv1(x)
        #print('out:',out.shape)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x


        return out

class FactorBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size, stride, downsample, pad, dilation):
        super(FactorBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, inplanes, kernel_size=(kernel_size, 1), stride = 1, pad=(1, pad), dilation=dilation),
                                    nn.ReLU(inplace=True),
                                    convbn(inplanes, planes, kernel_size=(1, kernel_size), stride=stride, pad=(pad,1), dilation=dilation),
                                    nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn(planes, planes, kernel_size=(kernel_size, 1), stride=1, pad=(1, pad), dilation=dilation),
                                  convbn(planes, planes, kernel_size=(1, kernel_size), stride=1, pad=(pad, 1), dilation=dilation))

        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left  = F.pad(torch.index_select(left,  3, Variable(torch.LongTensor([i for i in range(shift,width)])).cuda()),(shift,0,0,0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
        out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        
        self.layer0 = nn.Sequential(convbn(3, 32, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True))

        self.layer1_1 = self._make_layer(BasicBlock, 32, 3, 3, 2, 1,1,3/32)

        self.layer1_2 = self._make_layer(BasicBlock, 64, 8, 3, 2, 1,1,1)

        self.layer1_3 = self._make_layer(BasicBlock, 128, 3, 3, 2, 1,1,2)

        self.layer2_1 = self._make_layer(FactorBlock, 32, 1, 5, 2, 1,1,1)

        self.layer2_2 = self._make_layer(FactorBlock, 64, 3, 5, 2, 1,1,1)

        self.layer2_3 = self._make_layer(FactorBlock, 128, 1, 5, 2, 1,1,2)

        self.layer3_1 = self._make_layer(FactorBlock, 32, 1, 7, 2, 2,1,1)

        self.layer3_2 = self._make_layer(FactorBlock, 64, 3, 7, 2, 2,1,1)

        self.layer3_3 = self._make_layer(FactorBlock, 128, 1, 7, 2, 2,1,2)

        self.dilat1 = nn.Sequential(convbn(128, 32, 3, 1, 1,32),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))

        self.dilat2 = nn.Sequential(convbn(128, 32, 3, 1, 1,16),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))

        self.dilat3 = nn.Sequential(convbn(128, 32, 3, 1, 1,8),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 32, 3, 1, 1, 4),
                                      nn.ReLU(inplace=True))

        self.dilat4 = nn.Sequential(convbn(128, 32, 3, 1, 1,6),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))

        self.layer1_1_after = nn.Sequential(convbn(32, 32, 3, 2, 1, 1),
                                          nn.ReLU(inplace=True))


        self.lastconv = nn.Sequential(convbn(352, 128, 3,1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, kernel_size, stride, pad, dilation, order):
        downsample = None
        if block == BasicBlock:
          if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes*order, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

          layers = []
          layers.append(block(self.inplanes*order, planes, kernel_size, stride,downsample, pad, dilation))

          if blocks!=1:
            for i in range(1, blocks):
              layers.append(block(planes, planes, kernel_size, 1,None, pad, dilation))

        if block == FactorBlock:
          if stride!=1:
            downsample = nn.Sequential(convbn(self.inplanes*order, planes*block.expansion, kernel_size=(kernel_size, 1), stride=1, pad=(1, pad), dilation=dilation),
                                  convbn(planes*block.expansion, planes*block.expansion, kernel_size=(1, kernel_size), stride=2, pad=(pad, 1), dilation=dilation))

          layers = []
          layers.append(block(self.inplanes*order, planes, kernel_size, stride, downsample, pad, dilation))

          if blocks!=1:
            for i in range(1, blocks):
              layers.append(block(planes, planes, kernel_size, 1, None, pad, dilation))


        return nn.Sequential(*layers)


    def forward(self, x):
        out_0  = self.layer0(x)
        out1_1 = self.layer1_1(x)
        #print('out1_1:',out1_1.shape)
        #print('out_0:',out_0.shape)
        out2_1 = self.layer2_1(out_0)
        #print('out2_1:',out2_1.shape)
        out3_1 = self.layer3_1(out_0)
        out2_1 = out2_1 + out3_1
        out1_1 = out1_1 + out2_1

        out1_2 = self.layer1_2(out1_1)
        out2_2 = self.layer2_2(out2_1)
        out3_2 = self.layer3_2(out3_1)
        out2_2 = out2_2 + out3_2
        out1_2 = out1_2 + out2_2

        out1_3 = self.layer1_3(out1_2)
        out2_3 = self.layer2_3(out2_2)
        out3_3 = self.layer3_3(out3_2)
        out2_3 = out2_3 + out3_3
        out1_3 = out1_3 + out2_3

        out1_1 = self.layer1_1_after(out1_1)
        inPooling = F.upsample(out1_3, (out1_2.size()[2], out1_2.size()[3]),mode='bilinear')

        #Pooling 
        output_dilate1 = self.dilat1(inPooling)
        output_dilate2 = self.dilat2(inPooling)
        output_dilate3 = self.dilat3(inPooling)
        output_dilate4 = self.dilat4(inPooling)

        output_feature = torch.cat((out1_1, out1_2, inPooling, output_dilate1, output_dilate2, output_dilate3, output_dilate4), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature




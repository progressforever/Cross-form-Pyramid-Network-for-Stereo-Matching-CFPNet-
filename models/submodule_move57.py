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

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 3, 2, 1,1,1)

        self.layer2 = self._make_layer(BasicBlock, 64, 8, 3, 2, 1,1,1)

        self.layer3 = self._make_layer(BasicBlock, 128, 3, 3, 2, 1,1,2)

        self.layer1_after = nn.Sequential(convbn(32, 32, 3, 2, 1, 1),
                                          nn.ReLU(inplace=True))

        self.layer2_after = nn.Sequential(convbn(32, 64, 3, 2, 1, 1),
                                          nn.ReLU(inplace=True))

        self.layer3_after = nn.Sequential(convbn(64, 128, 3, 2, 1, 1),
                                          nn.ReLU(inplace=True))

        self.layer1_final = nn.Sequential(convbn(32, 32, 3, 2, 1, 1),
                                          nn.ReLU(inplace=True))




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


        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.concat_dilate_pool = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1,bias=False))

        self.lastconv = nn.Sequential(convbn(352, 128, 3,1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, kernel_size, stride, pad, dilation, order):
        downsample = None
        if stride != 1 :
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes* order, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes*order, planes, kernel_size, stride,downsample, pad, dilation))
        #print('zz')
        #self.inplanes = planes * block.expansion
        if blocks!=1:
          for i in range(1, blocks):
            layers.append(block(planes, planes, kernel_size, 1,None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        out_0  = self.layer0(x)

        out_1  = self.layer1(out_0)
        out_1_a  = self.layer1_after(out_0)
        out_1 = out_1 +out_1_a
        out_2  = self.layer2(out_1)
        out_2_a  = self.layer2_after(out_1)
        #print('out1:',out_1.shape)
        #print('out2:',out_2.shape)
        out_2  = out_2 + out_2_a
        
        out_3  = self.layer3(out_2)
        out_3_a = self.layer3_after(out_2)
        out_3  = out_3 + out_3_a
        #print('out_3:',out_3.shape)

        out_1 = self.layer1_final(out_1)

        
        inPooling = F.upsample(out_3, (out_2.size()[2], out_2.size()[3]),mode='bilinear')

        #Pooling 
        output_dilate1 = self.dilat1(inPooling)
        output_dilate2 = self.dilat2(inPooling)
        output_dilate3 = self.dilat3(inPooling)
        output_dilate4 = self.dilat4(inPooling)

        output_branch1 = self.branch1(inPooling)
        output_branch1 = F.upsample(output_branch1, (inPooling.size()[2],inPooling.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(inPooling)
        output_branch2 = F.upsample(output_branch2, (inPooling.size()[2],inPooling.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(inPooling)
        output_branch3 = F.upsample(output_branch3, (inPooling.size()[2],inPooling.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(inPooling)
        output_branch4 = F.upsample(output_branch4, (inPooling.size()[2],inPooling.size()[3]),mode='bilinear')

        #concat dilate and avgpool
        out_fusion1 = torch.cat((output_dilate1, output_branch1), 1)
        out_fusion1 = self.concat_dilate_pool(out_fusion1)

        out_fusion2 = torch.cat((output_dilate2, output_branch2), 1)
        out_fusion2 = self.concat_dilate_pool(out_fusion2)

        out_fusion3 = torch.cat((output_dilate3, output_branch3), 1)
        out_fusion3 = self.concat_dilate_pool(out_fusion3)

        out_fusion4 = torch.cat((output_dilate4, output_branch4), 1)
        out_fusion4 = self.concat_dilate_pool(out_fusion4)
        #print('out_fusion4:',out_fusion4.shape)


        #print('out1_1:',out1_1.shape)
        #print('out1_2:',out2_1.shape)
        #print('inPooling:',inPooling.shape)
        #print('out_fusion1:',out_fusion1.shape)
        #print('inPooling:',inPooling.shape)
        #print('out_fusion1:',out_fusion1.shape)
        output_feature = torch.cat((out_1, out_2, inPooling, out_fusion1, out_fusion2, out_fusion3, out_fusion4), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature




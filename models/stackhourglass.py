from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models import submodule as sb

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.inplanes = 32
        self.feature_extraction = sb.feature_extraction()

        self.dres0 = nn.Sequential(sb.convbn_3d(64, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(sb.convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    sb.convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = nn.Sequential(sb.convbn_3d(self.inplanes, self.inplanes, kernel_size=3, stride=2, pad=1),
                                    nn.ReLU(inplace=True),
                                    sb.convbn_3d(self.inplanes, self.inplanes, kernel_size=3, stride=1, pad=1))

        self.dres3 = nn.Sequential(sb.convbn_3d(self.inplanes, self.inplanes*2, kernel_size=3, stride=2, pad=1),
                                    nn.ReLU(inplace=True),
                                    sb.convbn_3d(self.inplanes*2, self.inplanes*2, kernel_size=3, stride=1, pad=1))

        self.dres4 = nn.Sequential(sb.convbn_3d(self.inplanes*2, self.inplanes*2, kernel_size=3, stride=2, pad=1),
                                    nn.ReLU(inplace=True),
                                    sb.convbn_3d(self.inplanes*2, self.inplanes*2, kernel_size=3, stride=1, pad=1))

        self.classif1 = nn.Sequential(sb.convbn_3d(128, 64, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(64, 32, 3, 1, 1))
        self.classif2 = nn.Sequential(sb.convbn_3d(128, 64, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(64, 32, 3, 1, 1))


        self.conv1_1 = nn.Sequential(sb.convbn_3d(32, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True))


        self.conv2_1 = nn.Sequential(sb.convbn_3d(32, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True))

        self.dres5 = nn.Sequential(sb.convbn_3d(self.inplanes*2, self.inplanes, kernel_size=3, stride=1, pad=1),
                                    nn.ReLU(inplace=True))

        self.deconv3_1 = nn.Sequential(nn.ConvTranspose3d(self.inplanes*2, self.inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(self.inplanes*2))

        self.deconv4_1 = nn.Sequential(nn.ConvTranspose3d(self.inplanes*2, self.inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(self.inplanes))

        self.deconv5_1 = nn.Sequential(nn.ConvTranspose3d(self.inplanes, self.inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(self.inplanes))

        self.deconv6_1 = nn.Sequential(nn.ConvTranspose3d(self.inplanes, self.inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(self.inplanes))

        self.deconv3_2 = nn.Sequential(nn.ConvTranspose3d(self.inplanes*2, self.inplanes*2, kernel_size=5, padding=2, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(self.inplanes*2))

        self.deconv4_2 = nn.Sequential(nn.ConvTranspose3d(self.inplanes*2, self.inplanes, kernel_size=5, padding=2, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(self.inplanes))

        self.deconv5_2 = nn.Sequential(nn.ConvTranspose3d(self.inplanes, self.inplanes, kernel_size=5, padding=2, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(self.inplanes))

        self.deconv6_2 = nn.Sequential(nn.ConvTranspose3d(self.inplanes, self.inplanes, kernel_size=5, padding=2, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(self.inplanes))


        self.deconv1 = nn.Sequential(nn.ConvTranspose3d(self.inplanes, self.inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(self.inplanes))


        self.deconv2 = nn.Sequential(nn.ConvTranspose3d(self.inplanes, 1, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):
        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(int(self.maxdisp/4)):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()
        #print('cost:',cost.shape)

        cost0 = self.dres0(cost)
        cost1 = self.dres1(cost0)
        cost1 = cost1+cost0

        cost2 = self.dres2(cost1)
        cost3 = self.dres3(cost2)
        cost4 = self.dres4(cost3)

        #print('cost1:',cost1.shape)
        F1_1 = self.conv1_1(cost1)

        #print('cost2:',cost2.shape)
        F1_2 = self.deconv5_1(cost2)

        #print('cost3:',cost3.shape)
        F1_3 = self.deconv4_1(cost3)
        F1_3 = self.deconv5_1(F1_3)


        #print('cost4:',cost4.shape)
        F1_4 = self.deconv3_1(cost4)
        F1_4 = self.deconv4_1(F1_4)
        F1_4 = self.deconv5_1(F1_4)

        F2_1 = self.conv2_1(cost1)

        F2_2 = self.deconv5_2(cost2)

        F2_3 = self.deconv4_2(cost3)
        F2_3 = self.deconv5_2(F2_3)

        F2_4 = self.deconv3_2(cost4)
        F2_4 = self.deconv4_2(F2_4)
        F2_4 = self.deconv5_2(F2_4)

        #print('F1_1:',F1_1.shape)
        #print('F1_2:',F1_2.shape)
        #print('F1_3:',F1_3.shape)
        #print('F1_4:',F1_4.shape)



        F_total1 = torch.cat((F1_1, F1_2, F1_3, F1_4), 1)
        F_total1 = self.classif1(F_total1)

        F_total2 = torch.cat((F2_1, F2_2, F2_3, F2_4), 1)
        F_total2 = self.classif2(F_total2)

        F_final = torch.cat((F_total1, F_total2), 1)

        #print('F_final:',F_final.shape)
        out1 = self.dres5(F_final)
        out2 = self.deconv1(out1)
        out3 = self.deconv2(out2)

        cost = F.upsample(out3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = sb.disparityregression(self.maxdisp)(pred)



        return pred


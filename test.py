#import torch
from torch.autograd import Variable

#input1 = Variable(torch.randn(1,32,256,512))

#conv1 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=7,stride=1,padding=3)

#conv2 = torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(7,1),stride=1,padding=(1,2))

#conv3 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(1,7),stride=1,padding=(2,1))

#output1 = conv1(input1)

#print("output1:",output1.shape)

#output2 = conv2(input1)
#print("output2_1:",output2.shape)

#output2 = conv3(output2)
#print("output2_2:",output2.shape)






#for i in range(1,9):
#	print(i)
#	if (i<3) or (i>3 and i<5) or (i>6 and i<8):
#		print("t")
#	else:
#		print("y")

#a = ['1','2','3']
#b = ['3','31','421']
#c = a+b
#print(c)
#import time
#print(time.strftime("%Y-%m-%d %X",time.localtime())+'  '+'epoch %d : learning rate = %d, mean training loss = %d, mean training error = %d (%d sec/epoch)' %(80, 1, 25, 12, 256))

import torch
import torch.nn.functional as F
import torch.nn as nn

input = Variable(torch.randn(64, 64, 64, 64))

print('input:',input.shape)

#branch1 = F.avg_pool3d(input,kernel_size=(16, 16, 16), stride=(16, 16, 16))

#output = branch1(input)

output = nn.Conv3d(input, 64, 32, 1, 15, 1)

print('output:',out.shape)
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *
from PIL import Image, ImageOps

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--savepath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        #print('output:',output.shape)
        pred_disp = output.data.cpu().numpy()
        #print('pred_disp:',pred_disp.shape)

        return pred_disp


def main():
   processed = preprocess.get_transform(augment=False)

   for inx in range(1):

       #imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
       #imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))

       #path_L='/home/frank/Documents/Program/Python_program/PSM-test/dingli/warped-00.jpg'
       #path_R='/home/frank/Documents/Program/Python_program/PSM-test/dingli/warped-01.jpg'

       path_L=args.datapath+'image_2/'+'%06d_10.png'%(inx)
       path_R=args.datapath+'image_3/'+'%06d_10.png'%(inx)
       
       #path_L=args.datapath+'colored_0/'+'%06d_10.png'%(inx)
       #path_R=args.datapath+'colored_1/'+'%06d_10.png'%(inx)

       imgL_o=Image.open(path_L).convert('RGB')
       imgR_o=Image.open(path_R).convert('RGB')
       w,h=imgL_o.size
       #print('w:',w)
       #print('h',h)
       #print('w=%d,h=%d'%(w,h))
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,h,w])
       imgR = np.reshape(imgR,[1,3,h,w])

       # pad to (384, 1248)
       top_pad = 384-h
       left_pad = 1248-w

       #top_pad = 544-h
       #left_pad = 960-w

       #print('top_pad:',top_pad)
       #print('left_pad:',left_pad)

       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       #print('imgR:',imgR.shape)

       #imgL = imgL[:256,:512,:]
       #imgR = imgR[:256,:512,:]

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))

       #top_pad   = 384-h
       #left_pad  = 1248-w

       #top_pad = 544-h
       #eft_pad = 960-w

       #img = pred_disp[:,:]

       img = pred_disp[top_pad:,:-left_pad]
       #print('final:',img.shape)
       path=args.savepath+'%06d_10.png'%(inx)

       #path = '/home/frank/Documents/Program/Python_program/PSM-test/dingli/00.png'


       skimage.io.imsave(path,(img*256).astype('uint16'))
       print('image %d has transformed'%(inx))

if __name__ == '__main__':
  main()

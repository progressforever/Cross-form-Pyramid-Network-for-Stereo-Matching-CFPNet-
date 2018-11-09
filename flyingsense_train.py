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
import numpy as np
import time
import math
from dataloader import SecenFlowLoader as SF
from dataloader import listflowfile as lf
from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lf.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         SF.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 1, shuffle= True, num_workers= 1, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         SF.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 1, shuffle= False, num_workers= 1, drop_last=False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('\n Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def cal_Acc(result,labels):
    res=np.array(result)
    size=res.shape[0]
    res=res.reshape(res.shape[0],res.shape[1],res.shape[2])

    groundTruth=np.array(labels)
    groundTruth=groundTruth.reshape(groundTruth.shape[0],groundTruth.shape[1],groundTruth.shape[2])

    total_acc=0
    for m in range(0,size):
        res_lin=res[m,:,:]
        groundTruth_lin=groundTruth[m,:,:]
        err=abs(res_lin-groundTruth_lin)
    
        num=0
        total=0
        for i in range(err.shape[0]):
            for j in range(err.shape[1]):
                if groundTruth_lin[i,j]!=0:
                    point=err[i,j]
                    if point>5 and point/groundTruth_lin[i,j]>0.05:
                        num=num+1
                    total=total+1
    
        acc=0
        if total!=0:
            acc=num/float(str(total))
        total_acc += acc

    acc=total_acc/size
    return acc

def train(imgL,imgR, disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            #output1, output2, output3 = model(imgL,imgR)
            #output1_s = torch.squeeze(output1,1)
            #output2_s = torch.squeeze(output2,1)
            #output3_s = torch.squeeze(output3,1)

            #loss = 0.5*F.smooth_l1_loss(output1_s[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2_s[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3_s[mask], disp_true[mask], size_average=True)

            output3 = model(imgL, imgR)
            output3 = torch.squeeze(output3, 1)
            loss = 2.5 * F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

        elif args.model == 'basic':
            output3 = model(imgL,imgR)
            loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data[0]
def cal_pred_train(imgL,imgR):
    model.eval()
    imgL=Variable(torch.FloatTensor(imgL))
    imgR=Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL,imgR=imgL.cuda(),imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL,imgR)

    output=torch.squeeze(output3.data.cpu(),1)[:,:,:]
    pred_disp=output.data.cpu().numpy()

    return pred_disp

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        #---------
        mask = disp_true < 192
        #----

        with torch.no_grad():
            output3 = model(imgL,imgR)

        output = torch.squeeze(output3.data.cpu(),1)[:,:,:]

        pred_disp = output.data.cpu().numpy()


        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error
        return loss, pred_disp

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    #print('learning_rate: ',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def main():

    start_full_time = time.time()
    print('iter:',len(TrainImgLoader))
    print('Continue 32 epoches')
    for epoch in range(1, args.epochs+1):
       #print('\n\n  epoch:: %d-th ,total_iter = 53 ' %(epoch))
       start_train_time = time.time()
       total_train_loss = 0
       total_train_err = 0
       lr = adjust_learning_rate(optimizer,epoch)

       ## training ##
       for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
         #start_time = time.time()

         loss = train(imgL_crop,imgR_crop, disp_crop_L)

         res=cal_pred_train(imgL_crop,imgR_crop)

         err=cal_Acc(res,disp_crop_L)
         #print('Iter %d training loss = %.3f ,err=%.3f, time = %.2f' %(batch_idx, loss, err, time.time() - start_time))
         total_train_loss += loss
         total_train_err += err
       print(time.strftime("%Y-%m-%d %X",time.localtime())+'   '+'epoch %d : learning rate = %f, mean training loss = %.2f, mean training error = %.5f (%.2f HR/epoch)' %(epoch+25, lr, total_train_loss/len(TrainImgLoader), total_train_err/len(TrainImgLoader), (time.time()-start_train_time)/3600))

       #SAVE
       savefilename = args.savemodel+'/checkpoint_'+str(epoch+25)+'.tar'
       torch.save({
            'epoch': epoch+25,
            'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
                    }, savefilename)

    print(time.strftime("%Y-%m-%d %X",time.localtime())+'   '+'full training time = %.2f HR \n\n\n training ends,testing begins now' %((time.time() - start_full_time)/3600))

    #------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    total_test_err=0
    strat_test_time = time.time()
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
           start_time=time.time()
           test_loss,res = test(imgL,imgR, disp_L)

           err=cal_Acc(res, disp_L)
           print('Iter %d test loss = %.5f, err=%.5f, time=%.2f HR' %(batch_idx, test_loss, err, (time.time()-start_time)/3600))
           total_test_loss += test_loss
           total_test_err += err


    print(time.strftime("%Y-%m-%d %X",time.localtime())+'   '+'mean test loss = %.5f, mean test err = %.5f, total_test_time = %.2f' %(total_test_loss/len(TestImgLoader), total_test_err/len(TestImgLoader), time.time()-strat_test_time))
    #----------------------------------------------------------------------------------
    #SAVE test information
    savefilename = args.savemodel+'testinformation.tar'
    torch.save({
            'test_loss': total_test_loss/len(TestImgLoader),
        }, savefilename)


if __name__ == '__main__':
   main()
    

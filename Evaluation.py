# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image


ACC_PIXEL = 3
RELATE_ERR = 0.05

# image's width
IMAGE_WIDTH = 416
# image's height
IMAGE_HEIGHT = 128
# image's width
IMAGE_ORG_WIDTH = 1248
# image's height
IMAGE_ORG_HEIGHT = 384

DEPTH_DIVIDING = 256.0


def Acc(result, labels):
    res = np.array(result, dtype=np.float32)/float(DEPTH_DIVIDING)
    res = res.reshape(res.shape[0], res.shape[1])

    groundTrue = np.array(labels, dtype=np.float32)/float(DEPTH_DIVIDING)
    groundTrue = groundTrue.reshape(groundTrue.shape[0], groundTrue.shape[1])
    #print('res:',res.shape)
    #print('ground:',groundTrue.shape)
    err = abs(res - groundTrue)

    num = 0
    errTotal = 0
    total = 0
    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            if groundTrue[i, j] != 0:     # this point is effect
                point = err[i, j]         # get the error
                errTotal = errTotal + point
                if point > ACC_PIXEL and point / groundTrue[i, j] > RELATE_ERR:
                    num = num + 1
                total = total + 1

    acc = 0
    if total != 0:
        acc = num / float(str(total))
        mas = errTotal / float(str(total))
        # diff = result - labels
        # num = np.sum(diff == True)
        # total = diff.shape[0] * diff.shape[1]
        # acc = num / float(str(total))
    return acc,mas


if __name__ == "__main__":
    total = 0
    num = 0
    errTotal = 0
    for i in range(200):
        imgPath = './resultImage/cfp/resultImage_2015_finutune_epoch495/%06d_10.png' % (i)

        groundPath = '/home1/Documents/Database/Kitti/testing/disp_occ_0/%06d_10.png' % (i)
#        if i % 5 != 0:
            #continue
        img = Image.open(imgPath)
        imgGround = Image.open(groundPath)

        acc,mas = Acc(img, imgGround)
        total = total + acc
        errTotal = mas + errTotal
        num = num + 1
        print (str(i) + ':' + str(acc) + ',' + str(mas))

    print ('total :%f,%f' % (total/num, errTotal/num)) 

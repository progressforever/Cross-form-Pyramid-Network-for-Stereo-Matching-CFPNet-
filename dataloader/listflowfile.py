import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

 classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
 #image = [img for img in classes if img.find('frames_cleanpass') > -1]
 #disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]
 all_left_img=[]
 all_right_img=[]
 all_left_disp = []
 test_left_img=[]
 test_right_img=[]
 test_left_disp = []

 #flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
 #flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
 flying_path = filepath + 'frames_cleanpass'
 flying_disp = filepath + 'disparity/'
 flying_dir = flying_path+'/TRAIN/'
 subdir = ['A','B','C']

 for ss in subdir:
    flying = os.listdir(flying_dir+ss)

    for ff in flying:
      imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
      for im in imm_l:
       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
         all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

       all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
         all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

 flying_dir = flying_path+'/TEST/'

 subdir = ['A','B','C']

 for ss in subdir:
    flying = os.listdir(flying_dir+ss)

    for ff in flying:
      imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
      for im in imm_l:
       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
         test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

       test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
         test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
    #test_rigth_img=test_right_img[:500]
    #test_left_img=test_left_img[:500]
    #test_left_disp=test_left_disp[:500]
 #imm_1 = os.listdir(flying_dir+'A/'+'0054'+'/left/')
 #for im in imm_1:
 # if is_image_file(flying_dir+'A/'+'0054'+'/left/'+im):
 #  test_left_img.append(flying_dir+'A/'+'0054'+'/left/'+im)
 # test_left_disp.append(flying_disp+'TEST/A/'+'0054'+'/left/'+im.split(".")[0]+'.pfm')
 # if is_image_file(flying_dir+'A/'+'0054'+'/right/'+im):
 #  test_right_img.append(flying_dir+'A/'+'0054'+'/right/'+im)

 return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


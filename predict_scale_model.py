
# Ver 1.14 add flip in test time

import os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from YCHOv329_HR_r4_tiny import Net
# from YCHOv329_HR_r4_tiny_SRD_x4 import Net
import time, math
import numpy as np
import h5py
from torchsummary import summary
import config
from torch_ssim import SSIM
import shutil
import dataset
import math

def print_(item):
    print(item)

from skimage.measure import compare_ssim as cal_ssim

import cv2

ens = False
CROP_ONCE = True
SAVE_IMG       = False
SAVE_MODEL     = True


FLIP_IN_TEST   = False
INFER_ONE_MORE = 0
CENTER_CROP = 256 #0
RESIZE      = 128
INTER_METHOD = cv2.INTER_CUBIC #cv2.INTER_CUBIC  cv2.INTER_LINEAR
USE_SR        = False             # Use SR model ( or just interpolation)
SR_SCALE      = int(CENTER_CROP/RESIZE)
DO_DERAIN_FIRST = 'INCORPORATED_DERAIN_SR' # 'INCORPORATED_DERAIN_SR' 'SR_AND_DERAIN' 'DERAIN_AND_SR'
BACK_ORIG_SCALE = True
EVAL_IN_RESIZE_SCALE= False
directory = './result_HR_r4_tiny_SR/rain200l_resize_{}/'.format(RESIZE)


# opt = parser.parse_args()

BATCH_SIZE=8
EPOCHS = 100
SR_RATE=[1,2,4]

SCALE_LABELS={256:1,128:2,64:3}
input_size = 256
NUM_CLASSES = len(SR_RATE)
FLIP  = True
AUTO_SR = True

train_set= dataset.Rain100H_SR( file_path='/data/derain_new/Rain100H/rain_data_train_Heavy', split = 'train', resize = input_size ,crop = True , original = False ,cutblur = False, flip = FLIP ,SR_RATE=SR_RATE ,AUTO_SR=AUTO_SR,return_scale=True)
test_set = dataset.Rain100H_SR(    file_path='/data/derain_new/Rain100H/rain_heavy_test'   ,  split = 'test', resize = input_size,crop = False , original = False ,SR_RATE=SR_RATE,AUTO_SR=AUTO_SR,return_scale=True)


training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=BATCH_SIZE,drop_last = True , shuffle=True)
valid_data_loader    = DataLoader(dataset=test_set , num_workers=0, batch_size=BATCH_SIZE,drop_last = False, shuffle=False)


from classification_models import squeezenet1_1
net = squeezenet1_1(num_classes =NUM_CLASSES)
net.to(torch.float)
if torch.cuda.is_available():
    print_('Use GPU')
    net.cuda()
net.train();
weight_decay = 0
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=weight_decay)
cross_entropy_loss = nn.CrossEntropyLoss()

def print_summary(summary):
        
        print_(' ')
        print_(summary['tag'])
        print_('epoch: {}      '.format(summary['epoch']     ))
        print_('total_num  : {}'.format(summary['total_num']  ))   
        print_('accuracy   : {}'.format(summary['accuracy']  ))   
def get_acc(data,loader,net,tag='test' , print_all=True):

    correct_num     = 0
    correct_num_in  = 0
    correct_num_out = 0
    pred_as_in_num  = 0
    pred_as_out_num = 0
    total_num       = 0
    for batch in loader: #Test
        with torch.no_grad():
            images,_,_,_,labels= batch
            # print('images shape',images.shape)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            labels_pred = net.forward(images)
            labels_pred = labels_pred.cpu().numpy()
            correct_num     += np.sum( np.argmax(labels_pred,axis=1) == labels.cpu().numpy() )
            total_num       += len(labels)
            
   
    summary    = {}
    summary['tag'       ] = tag 
    summary['epoch'     ] = epoch 
    summary['total_num' ] = total_num
    summary['accuracy'  ] = float(correct_num)     / total_num
    if print_all:
        print_summary(summary)

    return summary['accuracy'] , summary


best_acc = 0.0
for epoch in range(EPOCHS):
    train_loss_batch = 0
    net.train()
    for iteration, batch in enumerate(training_data_loader):
    
        input,_,_,_,scale = batch
        # print('scale',scale)

        images = input
        labels = scale
        
        # print('images shape',images.shape)

        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        labels_pred = net.forward(images)
        # backward
        loss = cross_entropy_loss(labels_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss = loss.data.item()
        train_loss_batch += train_loss 
    #cnt = cnt+1
    print_('train loss : {}'.format(train_loss_batch))
    print_(' ')
    #correct_num = 0
    #if epoch % 1 == 0:
    net.eval()
    #get_acc(train,train_loader,net,tag='train')
    valid_acc,valid_summary=get_acc(0,valid_data_loader,net,tag='valid',print_all=True)
    # test_acc ,test_summary =get_acc(test ,test_loader ,net,tag='test' ,print_all=False)
    if valid_acc >best_acc:
        if SAVE_MODEL:
            torch.save(net.state_dict(),'mia_weights/mia_{}_{}.npy'.format(ATTACK_ID,epoch))
        best_acc =valid_acc
        # best_valid_summary = valid_summary
        # # best_test_summary  = test_summary
        # print_('###################')
        # print_('best valid acc : {}'.format(best_acc))
        # get_acc(valid,valid_loader,net,tag='valid' ,print_all=True)
        # # get_acc(test ,test_loader ,net,tag='test'  ,print_all=True)
        # print_('###################')




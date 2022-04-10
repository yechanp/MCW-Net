

import dataset
global step

train_repo_dic ={
            'rain100h': 100, #100
            'rain100l': 100,
            'spadata' : 1000,#1000,
            'Rain1400': 1000,
            'Rain1200': 1000,
            'Rain800' : 100,
            'Cityscape':1000,
            'RainDrop': 100,

            }
eval_repo_dic ={
            'rain100h': 1000,
            'rain100l': 1000,
            'rain100h_old': 1000,
            'rain100l_old': 1000,
            'spadata': 10000,#10000,
            'Rain1400': 10000,
            'Rain1200': 10000,
            'Rain800' : 1000,
            'Cityscape':10000,
            'RainDrop':1000,

            }

#Learing Settings
batchSize = 4               #default: 4
nEpochs = 200 #150               #default: 150
start_epoch = 1             #default: 1
lr = 1e-4                   #default: 1e-4
input_size = 256


gpu = "1"                     #default: "0"

resume = ""
pretrained = ""             #default: ""

threads = 0                 #default: 0
cuda    = True                 #default: True
shuffle = True              #default: True

CUT_OUT  = True
FLIP     = True
#Dataset 
"""
train_dataset = rain100h, rain100l, spadata
test_dataset = rain100h, rain100l, rain100h_old, rain100l_old, spadata
"""
train_dataset = "rain100h"   #default = "rain100h"
test_dataset  = "rain100h"   #default = "rain100h"
eval_dataset  = "rain100h"   #default = "rain100h"

#Log
report_step = train_repo_dic[train_dataset]
eval_step = eval_repo_dic[test_dataset]

#Loss Function

att  = 0                     #default: 1 (Add Attention loss)
ssim = 0                     #default: 1 (Add SSIM loss)

att_alpha  = 1.00            #default: 0.01
ssim_alpha = 0.01            #default: 0.01

#Warm up
warmup = 0                  #default: 0

#Checkpoint folder
"""
It is just a base directory for pth file.
"train.py" will make additional folder in base directory
"""
checkpoint = "./checkpoint" #default: "./checkpoint"

#SGD Settings
use_sgd = False             #default: False (use adam)
momentum = 0.9              #default: 0.9
weight_decay = 1e-3         #default: 1e-3

#Learning rate Scheduling

lr = 1e-4
step = 100
nEpochs = 250
SAVE_EPOCHS = [100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,249,250]

if train_dataset in ["spadata"]:
    step = 1 
    nEpochs = 3
    SAVE_EPOCHS =[1,2,3]    
elif train_dataset in ["Rain1400","Rain1200","Cityscape"]:
    step = 30
    nEpochs = 100
    SAVE_EPOCHS = [10,20,30,40,50,60,70,80,90,100]
elif train_dataset in ['rain100h','rain100l','Rain800']:
    step = 100
    nEpochs = 250
    SAVE_EPOCHS = [100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,249,250]
elif train_dataset in ['RainDrop']:
    step = 200
    nEpochs = 500
    SAVE_EPOCHS = [100,150,200,250,300,350,400,425,450,460,470,480,490,499,500]
else:
    assert 1==2




########################################################################
def learning_rate_scheduling(epoch, lr):
    for i in range(epoch // step):
        lr = lr / 2
        # lr = lr / 10
    return lr#----------------------------------------------------Don't touch-------------------------------------------------#
train_data_dic = {
                'rain100h':             dataset.Rain100H(    file_path='/data/derain_new/Rain100H/rain_data_train_Heavy', split = 'train', resize = input_size ,crop = True , original = False ,    cutout = CUT_OUT, flip = FLIP),
                'rain100l':             dataset.Rain100L(    file_path='/data/derain_new/rain100L'                      , split = 'train', resize = input_size ,crop = True , original = False ,    cutout = CUT_OUT, flip = FLIP),
                'spadata':              dataset.Spadata(     file_path='/data/derain_new/SPANet/real_world_spanet.txt'  , split = 'train', resize = False      ,crop = False, original = True  ,    cutout = CUT_OUT, flip = FLIP),
                'rain1200':             dataset.Rain1200(    file_path='/data/derain/Rain1200/DID-MDN-training'         , split= 'train' , crop_size=input_size,crop = True ,cutout = CUT_OUT, flip = FLIP),
                'rain800' :             dataset.Rain800(                                resize=0       , split= 'train' , crop_size=input_size,crop = True ,                       cutout = CUT_OUT, flip = FLIP),
                'cityscape':            dataset.Cityscape(    file_path='/data/derain/cityscape'                        , split= 'train' , crop_size=input_size,crop = True ,                       cutout = CUT_OUT    ,  flip = FLIP    ),
                'raindrop' :             dataset.RainDrop(    file_path='/data/derain_new/raindrop_data'                , split= 'train' , crop_size=input_size,crop = True ,                       cutblur = CUT_OUT, flip = FLIP),


                
                
                }

test_data_dic = {
                'rain100h':             dataset.Rain100H(    file_path='/data/derain_new/Rain100H/rain_heavy_test',       split = 'test', resize = False,       crop = False , original = False),
                'rain100l':             dataset.Rain100L(    file_path='/data/derain_new/rain100L/rain_data_test_Light',  split = 'test', resize = False,       crop = False , original = False),
                'spadata':              dataset.Spadata(     file_path='/data/derain_new/SPANet/test/real_test_1000.txt', split = 'test', resize = False,       crop = False , original = True), #original : 512*512
                'rain100h_old':         dataset.Rain100H_old(file_path='/data/derain_new/Rain100H_old',                   split = 'test', resize = False,       crop = False , original = False),
                'rain100l_old':         dataset.Rain100L_old(file_path='/data/derain_new/rain100L/rain100L_old',          split = 'test', resize = False,       crop = False , original = False),
                'rain1200':             dataset.Rain1200(    file_path='/data/derain/Rain1200/DID-MDN-test'           ,   split = 'test', crop_size=input_size, crop = True                    ),
                'rain800' :             dataset.Rain800(                                                                  split = 'test', crop_size=input_size, crop = True                    ),
                'cityscape':            dataset.Cityscape(    file_path='/data/derain/cityscape'                    , split = 'test', crop_size=input_size, crop = True ,                  ),
                'raindrop' :             dataset.RainDrop(                                                                 split = 'test_a', crop_size=input_size, crop = True                    ),

                }


eval_data_dic = {
                'rain100h':             dataset.Rain100H(    file_path='/data/derain_new/Rain100H/rain_heavy_test',       split = 'test', resize = False,       crop = False , original = True),
                'rain100l':             dataset.Rain100L(    file_path='/data/derain_new/rain100L/rain_data_test_Light',  split = 'test', resize = False,       crop = False , original = True),
                'spadata':              dataset.Spadata(     file_path='/data/derain_new/SPANet/test/real_test_1000.txt', split = 'test', resize = False,       crop = False , original = True), #original : 512*512
                'rain100h_old':         dataset.Rain100H_old(file_path='/data/derain_new/Rain100H_old',                   split = 'test', resize = False,       crop = False , original = True),
                'rain100l_old':         dataset.Rain100L_old(file_path='/data/derain_new/rain100L/rain100L_old',          split = 'test', resize = False,       crop = False , original = True),
                'rain1200':             dataset.Rain1200(    file_path='/data/derain/Rain1200/DID-MDN-test'           ,   split = 'test', crop_size=None, crop = False                    ),
                'rain800' :             dataset.Rain800(                                                                  split = 'test', crop_size=None, crop = False                    ),
                'cityscape':            dataset.Cityscape(    file_path='/data/derain/cityscape'                    , split = 'test', crop = False ,                  ),
                'raindrop_a' :             dataset.RainDrop(                                            split = 'test_a', crop_size=None, crop = False                    ),
                'raindrop_b' :             dataset.RainDrop(                                            split = 'test_b', crop_size=None, crop = False                    ),

                }

train_dataset = train_dataset.lower()
test_dataset = test_dataset.lower()
eval_dataset = eval_dataset.lower()

train_set = train_data_dic[train_dataset]
test_set = test_data_dic[test_dataset]
eval_set = eval_data_dic[eval_dataset]



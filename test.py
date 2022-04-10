import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time
import dataset
import config_test as config
from skimage.measure import compare_ssim as cal_ssim
import cv2


parser = argparse.ArgumentParser(description="MCW Net")
parser.add_argument("--cuda", action="store_true", help="Use cuda? Default: True")
parser.add_argument("--model1", default=" ", type=str, help="Model path")
parser.add_argument("--gpu", default='1', help="GPU number to use when testing. Default: 0")
parser.add_argument("--result", default= './result', type=str, help="Result path Default: ./result")
parser.add_argument("--att", default=1, type=int, help="output include attention")
parser.add_argument("--use_gt", default=1, type=int, help="use gt or not")
parser.add_argument("--save_img", default=1, type=int, help="save image or not")
parser.add_argument("--model_name", default="MCW_Net_small", type=str, help="net name")

#=================================================================================================================
# model_name = "MCW_Net_small"
model_name = "MCW_Net_small"
# test_dataset = config.test_dataset.lower() # set in config
#=================================================================================================================


opt = parser.parse_args()
# model_name = opt.model_name
# module    = __import__(model_name)
# model1       = module.Net
CROP_ONCE = False
SAVE_IMG  = opt.save_img
USE_GT    = opt.use_gt



def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr
    
cuda = True
opt.cuda = True

print('opt' ,opt)

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


model1 = torch.load(opt.model1, map_location=lambda storage, loc: storage)["model"]
model1 = model1.cuda()


avg_ssim_noisy = 0.0
avg_ssim_predicted = 0.0
avg_ssim_ens_predicted = 0.0
avg_psnr_noisy = 0.0
avg_psnr_predicted = 0.0
avg_psnr_ens_predicted = 0.0
avg_elapsed_time = 0.0
count = 0.0
flag = 0

eval_data = config.eval_set

directory = os.path.join("result_"+model_name, config.eval_dataset)


def infer_image(input,MAX_SIZE=2048,CROP_SIZE=512):
    input        = torch.from_numpy(input.copy())
    [tc, th, tw] = input.shape

    
    if th % 8 == 0 and  tw % 8 == 0 :

        input_croped = input
        input_croped = Variable(input_croped).view(1,tc,th, tw)
        if cuda:
            input_croped = input_croped.cuda()
          
        start_time = time.time()
        with torch.no_grad():
            if True:
                attention, output = model1(input_croped)
                attention = np.squeeze(attention.cpu().detach().numpy().astype(np.float32))
                attention = attention.transpose((1,2,0))

            output = np.squeeze(output.cpu().detach().numpy().astype(np.float32))
            
        canvas = output
        return canvas
    
            
    
    
    if max(th,tw) > 800 and min(th,tw)>512:
        num_h  = int((th-CROP_SIZE)/(CROP_SIZE-32) +1 ) +1
        num_w  = int((tw-CROP_SIZE)/(CROP_SIZE-32) +1 ) +1
        canvas = np.zeros_like(input)
        patched_area    = np.zeros_like(canvas,dtype=np.int16)
        strip_area      = np.zeros_like(canvas,dtype=np.int8)
        square_area     = np.zeros_like(canvas,dtype=np.int8)
        
        
        for a in range(0,(CROP_SIZE-32)*(num_h),CROP_SIZE-32):
            for b in range(0,(CROP_SIZE-32)*(num_w),CROP_SIZE-32):
                if not a+CROP_SIZE >= th:
                    a_start  = a
                else:
                    a_start = th -CROP_SIZE
                if not b+CROP_SIZE >= tw:
                    b_start  = b
                else:
                    b_start = tw -CROP_SIZE
                input_croped = input[:,a_start:a_start+CROP_SIZE,b_start:b_start+CROP_SIZE]
                patched_area[        :,a_start:a_start+CROP_SIZE,b_start:b_start+CROP_SIZE]+=1
                input_croped = Variable(input_croped).view(1,tc,CROP_SIZE,CROP_SIZE)

                if cuda:
                    input_croped = input_croped.cuda()
                start_time = time.time()
                with torch.no_grad():
                    if True:
                        attention, output = model1(input_croped)
                        attention = np.squeeze(attention.cpu().detach().numpy().astype(np.float32))
                        attention = attention.transpose((1,2,0))

                    output = np.squeeze(output.cpu().detach().numpy().astype(np.float32))                        
                             
                canvas[:,a_start:a_start+CROP_SIZE,b_start:b_start+CROP_SIZE] += output
        patched_area = patched_area.astype(np.float32)
        canvas = canvas / patched_area
    
    else:
        rh = th -   ( int( (th-1)//32)* 32 )
        rw = tw -   ( int( (tw-1)//32)* 32 )
        # print('rh',rh,rw)
    
        canvas = np.zeros_like(input)
        strip_area    = np.zeros_like(canvas,dtype=np.int16)
        square_area   = np.zeros_like(canvas,dtype=np.int16)

        square_area[:,rh:-rh,rw:-rw] += 1
        strip_area[:,rh:-rh,:] += 1
        strip_area[:,:,rw:-rw] += 1
        square_area = square_area.astype(np.bool)
        strip_area  = strip_area.astype(np.bool)

   

        for a in [0,rh]:
            for b in [0,rw]:
      
                input_croped = input[:,a:th+a-rh,b:tw+b-rw]
                input_croped = Variable(input_croped).view(1,tc,th-rh, tw-rw)
                if cuda:
                    input_croped = input_croped.cuda()
                  
                start_time = time.time()
                with torch.no_grad():
                    if opt.att:
                        attention, output = model1(input_croped)
                        attention = np.squeeze(attention.cpu().detach().numpy().astype(np.float32))
                        attention = attention.transpose((1,2,0))
                    else:
                        output = model1(input_croped)
                    output = np.squeeze(output.cpu().detach().numpy().astype(np.float32))

                canvas[:,a:th+a-rh,b:tw+b-rw] += output
           
        if True:
            canvas[square_area] = canvas[square_area]/2
            canvas[strip_area]  = canvas[strip_area]/2

    return canvas


    
model1.eval()
for i in range(eval_data.__len__()):

    
    print(i)
    input, gt = eval_data[i]
    input     = input.cpu().detach().numpy()
    [tc, th, tw] = input.shape
    print('input.shape',input.shape)
    canvas = infer_image(input)
    canvas_flip = infer_image(input[:,:,::-1])
    canvas      += canvas_flip[:,:,::-1]
    canvas      = canvas/2
        

  
    gt        = gt.cpu().detach().numpy()
    input     = input.astype(np.float32)
    input     = input.transpose((1,2,0))
    gt        = gt.transpose((1,2,0))
    canvas    = canvas.transpose((1,2,0))

    psnr_predicted = output_psnr_mse(canvas, gt)
    avg_psnr_predicted += psnr_predicted
   

    ssim_ens_predicted = cal_ssim(canvas, gt, gaussian_weights=True, use_sample_covariance=False, multichannel=True , sigma= 1.5 , data_range= 1.0)
    avg_ssim_ens_predicted += ssim_ens_predicted
    
    print('psnr_predicted',psnr_predicted)
    print('ssim_ens_predicted',ssim_ens_predicted)
   

    name = eval_data.rains[i]
    output = canvas
    
    
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if SAVE_IMG:
        
        if '/' in name:
            name = os.path.split(name)[-1]
        name = os.path.splitext(name)[0] + '.png'
            
        path = os.path.join(directory , name )
        print('save path',path)

        done = cv2.imwrite(path, np.hstack([output*255])[:,:,::-1] )
    
        
   

time_per_patch = avg_elapsed_time / eval_data.__len__()
avg_psnr_predicted /= eval_data.__len__()

avg_ssim_ens_predicted /= eval_data.__len__()
print("avg_psnr_predicted : {}".format(avg_psnr_predicted))
print("avg_ssim_predicted : {}".format(avg_ssim_ens_predicted))



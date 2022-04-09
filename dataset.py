import torch.utils.data as data
import torch
import h5py
import cv2
import os
import numpy as np
import math

class Rain100H_old(data.Dataset):
    def __init__(self, file_path='/data/derain_new/Rain100H_old', split = 'test', resize = False, crop = False, original = False):
        super(Rain100H_old, self).__init__()
        self.rain_path = os.path.join(file_path, 'rainy')
        self.norain_path = os.path.join(file_path, 'norain')
        self.rains = os.listdir(self.rain_path)
        # self.rains = [i for i in self.all_name if i.startswith('rain')]
        self.original = original

    def __getitem__(self, index):
        self.data = cv2.imread(os.path.join(self.rain_path,self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        self.target = cv2.imread(os.path.join(self.norain_path,'no'+self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        c, h, w = self.data.shape
        if not self.original:
            if w%2 == 1:
                w = w-1
            if h%2 == 1:
                h = h-1
        self.data = self.data[:,:h,:w]
        self.target = self.target[:,:h,:w]
        return torch.from_numpy(self.data.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)

class Rain100L_old(data.Dataset): 
    def __init__(self, file_path='/data/derain_new/rain100L/rain100L_old', split = 'test', resize = False, crop = False, original = False):
        super(Rain100L_old, self).__init__()
        self.rain_path = os.path.join(file_path, 'rainy')
        self.norain_path = os.path.join(file_path, 'norain')
        self.rains = os.listdir(self.rain_path)
        self.original = original

    def __getitem__(self, index):
        self.data = cv2.imread(os.path.join(self.rain_path, self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        self.target = cv2.imread(os.path.join(self.norain_path,'no'+self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        c, h, w = self.data.shape
        if not self.original:
            if w%2 == 1:
                w = w-1
            if h%2 == 1:
                h = h-1
        self.data = self.data[:,:h,:w]
        self.target = self.target[:,:h,:w]
        return torch.from_numpy(self.data.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)

class Rain100H(data.Dataset): 
    def __init__(self, file_path='/data/derain_new/Rain100H/rain_data_train_Heavy', split = 'train', resize = 256, crop = True , original = False, cutout= False, flip = False):
        super(Rain100H, self).__init__()
        self.path = file_path
        self.all_name = os.listdir(file_path)
        self.rains = [i for i in self.all_name if i.startswith('rain')]
        self.resize = resize
        self.crop = crop
        self.split = split
        self.original  = original
        self.cutout = cutout
        self.flip = flip
    def __getitem__(self, index):

        if self.split == 'train':     
            if self.crop:          
                self.data = cv2.imread(os.path.join(self.path,self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
                self.target = cv2.imread(os.path.join(self.path,'no'+self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
                c, h, w = self.data.shape
                ws = np.random.randint(w-self.resize)
                hs = np.random.randint(h-self.resize)
                self.data = self.data[:, hs:hs+self.resize, ws:ws+self.resize]
                self.target = self.target[:, hs:hs+self.resize, ws:ws+self.resize]
                if self.flip:
                    if np.random.random() > 0.5:
                        self.data   = self.data[:,:,::-1]
                        self.target = self.target[:,:,::-1]
                if self.cutout:
                    if np.random.random() > 0.5:
                        cut_ratio = np.random.rand()/2
                        ch, cw = np.int(self.resize*cut_ratio), np.int(self.resize*cut_ratio)
                        cy = np.random.randint(0, self.resize-ch+1)
                        cx = np.random.randint(0, self.resize-cw+1)
                        self.data[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
            else:
                self.data = cv2.resize(cv2.imread(os.path.join(self.path,self.rains[index])), (self.resize,self.resize))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
                self.target = cv2.resize(cv2.imread(os.path.join(self.path,'no'+self.rains[index])), (self.resize,self.resize))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        else:
            self.data = cv2.imread(os.path.join(self.path,self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            self.target = cv2.imread(os.path.join(self.path,'no'+self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            c, h, w= self.data.shape
            if not self.original:
                if w%2 == 1:
                    w = w-1
                if h%2 == 1:
                    h = h-1
            self.data = self.data[:,:h,:w]
            self.target = self.target[:,:h,:w]
        return torch.from_numpy(self.data.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)

class Rain100L(data.Dataset): 
    def __init__(self, file_path='/data/derain_new/rain100L', split = 'train', resize = 256, crop = True , original = False , cutout= False, flip = False):
        super(Rain100L, self).__init__()
        self.path = file_path
        if split == 'train':        
            self.all_name = os.listdir(os.path.join(file_path,'rain'))
            self.rains = [i[:-4] for i in self.all_name if i.startswith('norain')]
        else:
            self.all_name = os.listdir(os.path.join(file_path,'rain','X2'))
            self.rains = [i[:-4] for i in self.all_name if i.startswith('norain')]
        self.resize = resize
        self.crop = crop
        self.split = split
        self.original  = original
        self.cutout = cutout
        self.flip = flip

    def __getitem__(self, index):
        if self.split == 'train':
            if self.crop:
                self.data = cv2.imread(os.path.join(self.path,'rain',self.rains[index]+'.png'))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
                self.target = cv2.imread(os.path.join(self.path,'norain',self.rains[index][:-2]+'.png'))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
                c, h, w = self.data.shape
                ws = np.random.randint(w-self.resize)
                hs = np.random.randint(h-self.resize)
                self.data = self.data[:, hs:hs+self.resize, ws:ws+self.resize]
                self.target = self.target[:, hs:hs+self.resize, ws:ws+self.resize]
                if self.flip:
                    if np.random.random() > 0.5:
                        self.data   = self.data[:,:,::-1]
                        self.target = self.target[:,:,::-1]
                if self.cutout:
                    if np.random.random() > 0.5:
                        cut_ratio = np.random.rand()/2
                        ch, cw = np.int(self.resize*cut_ratio), np.int(self.resize*cut_ratio)
                        cy = np.random.randint(0, self.resize-ch+1)
                        cx = np.random.randint(0, self.resize-cw+1)
                        self.data[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
                
            else:
                self.data = cv2.imread(os.path.join(self.path,'rain',self.rains[index]+'.png'))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
                self.target = cv2.imread(os.path.join(self.path,'norain',self.rains[index][:-2]+'.png'))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        else:
            self.data = cv2.imread(os.path.join(self.path,'rain','X2',self.rains[index]+'.png'))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            self.target = cv2.imread(os.path.join(self.path,'norain',self.rains[index][:-2]+'.png'))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            c, h, w = self.data.shape
            if not self.original:
                if w%2 == 1:
                    w = w-1
                if h%2 == 1:
                    h = h-1
            self.data = self.data[:,:h,:w]
            self.target = self.target[:,:h,:w]
        return torch.from_numpy(self.data.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)

class Spadata(data.Dataset): 

    def __init__(self, file_path='/data/derain_new/SPANet/real_world_spanet.txt', split = 'train', resize = False, crop = False, original = True , cutout= False, flip = False):
        super(Spadata, self).__init__()
        self.file_path = file_path
        slash_index = [i for i in range(len(file_path)) if file_path[i] == '/']
        self.path = file_path[:slash_index[-1]]
        self.all_name = np.loadtxt(file_path, delimiter = ' ', dtype = 'str')
        self.cutout = cutout
        self.flip = flip
        
        self.rains = self.all_name[:,0]
        


    def __getitem__(self, index):
        self.data = cv2.imread(os.path.join(self.path, self.all_name[index][0][1:]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        self.target = cv2.imread(os.path.join(self.path, self.all_name[index][1][1:]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        if self.flip:
            if np.random.random() > 0.5:
                self.data   = self.data[:,:,::-1]
                self.target = self.target[:,:,::-1]
        if self.cutout:
            if np.random.random() > 0.5:
                cut_ratio = np.random.rand()/2 
                img_size = self.data.shape[1]
                ch, cw = np.int(img_size*cut_ratio), np.int(img_size*cut_ratio)
                cy = np.random.randint(0, img_size-ch+1)
                cx = np.random.randint(0, img_size-cw+1)
                self.data[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
        
        
        return torch.from_numpy(self.data.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.all_name[:,0])
          
class Rain1400(data.Dataset): 

    def __init__(self, file_path='/data/derain/Rain1400/', split = 'train', crop = True ,crop_size = 384 , cutout= False):
        super(Rain1400, self).__init__()
        self.path = file_path
        self.all_name = os.listdir(file_path)
        self.rains = [i for i in self.all_name if i.startswith('rain')]
        self.crop_size = crop_size
        self.crop = crop
        self.split = split
        self.cutout = cutout

        
        if split == 'train':
            train_foler = os.path.join('train', 'Rain12600','rainy_image')
            self.rains = [ os.path.join(train_foler,item)  for item in  os.listdir(os.path.join(file_path,train_foler))]
            
        elif split == 'test':
            test_foler = os.path.join('test','rainy_image')
            self.rains = [ os.path.join(test_foler,item) for item in  os.listdir(os.path.join(file_path,test_foler))]

    def __getitem__(self, index):

        
        if self.split == 'train':
            target_path = os.path.join(self.path,self.rains[index].replace('rainy_image','groundtruth').split('_')[0]+'.jpg').replace('groundtruth','ground_truth')
            
            self.data   = cv2.imread(os.path.join(self.path,self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            self.target = cv2.imread(target_path)[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0

            if self.crop:
                
                c, h, w = self.data.shape
                hs = np.random.randint(h-self.crop_size)
                ws = np.random.randint(w-self.crop_size)
                self.data   = self.data[:  , hs:hs+self.crop_size, ws:ws+self.crop_size]
                self.target = self.target[:, hs:hs+self.crop_size, ws:ws+self.crop_size]
                if self.cutout:
                    if np.random.random() > 0.5:
                        cut_ratio = np.random.rand()/2
                        ch, cw = np.int(self.crop_size*cut_ratio), np.int(self.crop_size*cut_ratio)
                        cy = np.random.randint(0, self.crop_size-ch+1)
                        cx = np.random.randint(0, self.crop_size-cw+1)
                        self.data[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
            else:
                assert 1==2
        else:
            target_path = os.path.join(self.path,self.rains[index].replace('rainy_image','groundtruth').split('_')[0]+'.jpg').replace('groundtruth','ground_truth')
            
            self.data   = cv2.imread(os.path.join(self.path,self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            self.target = cv2.imread(target_path)[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            if self.crop:
                
                c, h, w = self.data.shape
                hs = np.random.randint(h-self.crop_size)
                ws = np.random.randint(w-self.crop_size)
                self.data   = self.data[:  , hs:hs+self.crop_size, ws:ws+self.crop_size]
                self.target = self.target[:, hs:hs+self.crop_size, ws:ws+self.crop_size]
                
                if self.cutout:
                    if np.random.random() > 0.5:
                        cut_ratio = np.random.rand()/2
                        ch, cw = np.int(self.crop_size*cut_ratio), np.int(self.crop_size*cut_ratio)
                        cy = np.random.randint(0, self.crop_size-ch+1)
                        cx = np.random.randint(0, self.crop_size-cw+1)
                        self.data[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
            else:
                pass
            
        return torch.from_numpy(self.data.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)

class Rain1200(data.Dataset):
    def __init__(self, file_path='/data/derain/Rain1200/DID-MDN-training' ,split='train', crop = True ,crop_size = 256,cutout= False , only_medium=False , flip = False):
        super(Rain1200, self).__init__()
        self.file_path = file_path
        folders = os.listdir(file_path)
        if only_medium:
            folders = [item for item in folders if 'Medium' in item]
        self.rains = []
        folder_file_lists = [ ]
        if split == 'train':
            for folder in folders:
                self.rains
                folder_file_list = [os.path.join(folder,'train2018new' , item) for item in os.listdir(os.path.join(file_path,folder,'train2018new'))]
                folder_file_lists.append(folder_file_list)
            for i in range( len( folder_file_lists[0] ) ):
                if only_medium:
                    self.rains.append( folder_file_lists[0][i])
                else:
                    self.rains.append( folder_file_lists[0][i])
                    self.rains.append( folder_file_lists[1][i])
                    self.rains.append( folder_file_lists[2][i])
            print( 'len of rains is : {}'.format(len(self.rains)))

        elif split == 'test':
            self.rains = os.listdir(file_path)
        
        
        self.crop_size = crop_size
        self.crop      = crop
        self.cutout = cutout
        self.flip = flip


    def __getitem__(self, index):
        whole   = cv2.imread(os.path.join(self.file_path , self.rains[index] ) )[:,:,::-1]
        self.rain    = whole[:,:512,:].transpose((2,0,1)).astype(np.float32)/255.0
        self.target  = whole[:,512:,:].transpose((2,0,1)).astype(np.float32)/255.0

        if self.crop:
            c, w, h = self.rain.shape
            ws = np.random.randint(w-self.crop_size)
            hs = np.random.randint(h-self.crop_size)
            self.rain   = self.rain[:  , ws:ws+self.crop_size, hs:hs+self.crop_size]
            self.target = self.target[:, ws:ws+self.crop_size, hs:hs+self.crop_size]
            self.rain = self.rain[:,:w,:h]
            self.target = self.target[:,:w,:h]
            if self.flip:
                if np.random.random() > 0.5:
                    self.rain   = self.rain[:,:,::-1]
                    self.target = self.target[:,:,::-1]
            if self.cutout:
                if np.random.random() > 0.5:
                    cut_ratio = np.random.rand()/2
                    ch, cw = np.int(self.crop_size*cut_ratio), np.int(self.crop_size*cut_ratio)
                    cy = np.random.randint(0, self.crop_size-ch+1)
                    cx = np.random.randint(0, self.crop_size-cw+1)
                    self.rain[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
            
       
        
        return torch.from_numpy(self.rain.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)

class Rain800(data.Dataset): 
    def __init__(self, file_path='/data/derain/Rain800' ,split='train', crop = True ,crop_size = 256 , cutout= False , resize = 0 , flip = False):
        super(Rain800, self).__init__()
        self.file_path = file_path
        self.split     = split
        
        self.crop_size = crop_size
        self.crop      = crop
        self.cutout = cutout
        self.resize = resize 
        self.flip = flip


        folders = os.listdir(file_path)
        self.rains = []
        if split == 'train':
            self.rains = [ 'train/'+ item for item in  os.listdir(os.path.join(file_path,'train'))]
            
        elif split == 'test':
            self.rains = [ 'val/'+ item for item in  os.listdir(os.path.join(file_path,'val'))]
            self.rains = sorted(self.rains)
        
       

    def __getitem__(self, index):

        whole   = cv2.imread(os.path.join(self.file_path , self.rains[index] ) )[:,:,::-1]
        half = int( whole.shape[1] //2 )
        self.target  = whole[:,:half,:]
        self.rain    = whole[:,half:,:]
        
        if self.crop:
            h,w,c = self.target.shape
            if self.resize > 0 :
                if min(h,w) < self.crop_size:
                    ratio  =  float(self.crop_size+1) / min(h,w)
                    self.target = cv2.resize( self.target, ( int(w*ratio) , int(h*ratio) ) ).transpose((2,0,1)).astype(np.float32)/255.0
                    self.rain   = cv2.resize( self.rain,   ( int(w*ratio) , int(h*ratio) ) ).transpose((2,0,1)).astype(np.float32)/255.0
                else:
                    ratio  =  float(self.resize) / min(h,w)
                    self.target = cv2.resize( self.target, ( int(w*ratio) , int(h*ratio) ) ).transpose((2,0,1)).astype(np.float32)/255.0
                    self.rain   = cv2.resize( self.rain,   ( int(w*ratio) , int(h*ratio) ) ).transpose((2,0,1)).astype(np.float32)/255.0
            else:
                if min(h,w) < self.crop_size:
                    ratio  =  float(self.crop_size+1) / min(h,w)
                    self.target = cv2.resize( self.target, ( int(w*ratio) , int(h*ratio) ) ).transpose((2,0,1)).astype(np.float32)/255.0
                    self.rain   = cv2.resize( self.rain,   ( int(w*ratio) , int(h*ratio) ) ).transpose((2,0,1)).astype(np.float32)/255.0
                else:
                    self.target = self.target.transpose((2,0,1)).astype(np.float32)/255.0
                    self.rain   = self.rain.transpose((2,0,1)).astype(np.float32)/255.0           
        else:
            self.target = self.target.transpose((2,0,1)).astype(np.float32)/255.0
            self.rain   = self.rain.transpose((2,0,1)).astype(np.float32)/255.0          

        if self.crop:
            c, h, w = self.rain.shape
            crop_size = self.crop_size
            
            ws = np.random.randint(w-crop_size)
            hs = np.random.randint(h-crop_size)
            self.rain   = self.rain[:  , hs:hs+crop_size, ws:ws+crop_size]
            self.target = self.target[:, hs:hs+crop_size, ws:ws+crop_size]
            self.rain   = self.rain[:,:h,:w]
            self.target = self.target[:,:h,:w]
            if self.flip:
                if np.random.random() > 0.5:
                    self.rain   = self.rain[:,:,::-1]
                    self.target = self.target[:,:,::-1]
            if self.cutout:     
                if np.random.random() > 0.5:
                    cut_ratio = np.random.rand()/2
                    ch, cw = np.int(self.crop_size*cut_ratio), np.int(self.crop_size*cut_ratio)
                    cy = np.random.randint(0, self.crop_size-ch+1)
                    cx = np.random.randint(0, self.crop_size-cw+1)
                    self.rain[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
            
      
        return torch.from_numpy(self.rain.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)

class Cityscape(data.Dataset):
    def __init__(self, file_path='/data/derain/cityscape', split = 'train', crop = True ,crop_size = 256 ,cutout= False, flip = False):
        super(Cityscape, self).__init__()
        self.path = file_path
        self.all_name = os.listdir(file_path)
        self.rains = [i for i in self.all_name if i.startswith('rain')]
        self.crop_size = crop_size
        self.crop = crop
        self.split = split
        self.cutout = cutout
        self.flip = flip

        
        folders = os.listdir(file_path)
        self.rains = []
        if split == 'train':
            folders = os.listdir(os.path.join(file_path,'leftImg8bit_rain','train'))
            for folder in folders:
                self.rains  += [os.path.join('leftImg8bit_rain','train',folder, item) for item in os.listdir(os.path.join(file_path,'leftImg8bit_rain','train',folder))]
                
        if split == 'test':
            folders = os.listdir(os.path.join(file_path,'leftImg8bit_rain','val'))
            for folder in folders:
                self.rains  += [os.path.join('leftImg8bit_rain','val',folder, item) for item in os.listdir(os.path.join(file_path,'leftImg8bit_rain','val',folder))]
        
            
       

    def __getitem__(self, index):

        
        if self.split == 'train':
            target_path = os.path.join(self.path,self.rains[index].replace('leftImg8bit_rain','leftImg8bit').split('_alpha')[0]+'.png')
            
            self.data   = cv2.imread(os.path.join(self.path,self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            self.target = cv2.imread(target_path)[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0

            if self.crop:
                
                c, h, w = self.data.shape
                hs = np.random.randint(h-self.crop_size)
                ws = np.random.randint(w-self.crop_size)
                self.data   = self.data[:  , hs:hs+self.crop_size, ws:ws+self.crop_size]
                self.target = self.target[:, hs:hs+self.crop_size, ws:ws+self.crop_size]
                
                if self.flip:
                    if np.random.random() > 0.5:
                        self.data   = self.data[:,:,::-1]
                        self.target = self.target[:,:,::-1]
                if self.cutout:
                    if np.random.random() > 0.5:
                        cut_ratio = np.random.rand()/2
                        ch, cw = np.int(self.crop_size*cut_ratio), np.int(self.crop_size*cut_ratio)
                        cy = np.random.randint(0, self.crop_size-ch+1)
                        cx = np.random.randint(0, self.crop_size-cw+1)
                        self.data[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
                    
            else:
                assert 1==2
        else:
            target_path = os.path.join(self.path,self.rains[index].replace('leftImg8bit_rain','leftImg8bit').split('_alpha')[0]+'.png')
            
            self.data   = cv2.imread(os.path.join(self.path,self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            self.target = cv2.imread(target_path)[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            if self.crop:
                
                c, h, w = self.data.shape
                hs = np.random.randint(h-self.crop_size)
                ws = np.random.randint(w-self.crop_size)
                self.data   = self.data[:  , hs:hs+self.crop_size, ws:ws+self.crop_size]
                self.target = self.target[:, hs:hs+self.crop_size, ws:ws+self.crop_size]
            else:
                pass
                #assert 1==2
            
        return torch.from_numpy(self.data.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)

class RainDrop(data.Dataset):
    def __init__(self, file_path='/data/derain_new/raindrop_data/' ,split='train', crop = True ,crop_size = 256,cutblur= False , flip = False):
        super(RainDrop, self).__init__()
        self.file_path = file_path
        folders = os.listdir(file_path)
        self.rains = []
        folder_file_lists = [ ]
        if split == 'train':
            self.rains = [os.path.join(split,'data',item) for item in os.listdir(os.path.join(file_path,split,'data'))]
            self.gts   = [item.replace('data','gt').replace('_rain','_clean') for item in self.rains]
            print( 'len of rains is : {}'.format(len(self.rains)))

        else:
            self.rains = [os.path.join(split,'data',item) for item in os.listdir(os.path.join(file_path,split,'data'))]
            self.gts   = [item.replace('data','gt').replace('_rain','_clean') for item in self.rains]
        
        self.crop_size = crop_size
        self.crop      = crop
        self.cutblur = cutblur
        self.flip = flip
        self.split = split


        # self.rains = [i for i in self.all_name if i.startswith('rain')]

    def __getitem__(self, index):

        
        rain_path   = os.path.join(self.file_path,self.rains[index])
        target_path = os.path.join(self.file_path,self.gts[index])


        self.rain    = cv2.imread(rain_path)[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        self.target  = cv2.imread(target_path)[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
        

        if self.crop:
            c, w, h = self.rain.shape
            ws = np.random.randint(w-self.crop_size)
            hs = np.random.randint(h-self.crop_size)
            if 'test' in self.split:
                ws,hs = 0 , 0
            self.rain   = self.rain[:  , ws:ws+self.crop_size, hs:hs+self.crop_size]
            self.target = self.target[:, ws:ws+self.crop_size, hs:hs+self.crop_size]
            self.rain = self.rain[:,:w,:h]
            self.target = self.target[:,:w,:h]
            if self.flip:
                if np.random.random() > 0.5:
                    self.rain   = self.rain[:,:,::-1]
                    self.target = self.target[:,:,::-1]
            if self.cutblur:
                if np.random.random() > 0.5:
                    cut_ratio = np.random.rand()/2
                    ch, cw = np.int(self.crop_size*cut_ratio), np.int(self.crop_size*cut_ratio)
                    cy = np.random.randint(0, self.crop_size-ch+1)
                    cx = np.random.randint(0, self.crop_size-cw+1)
                    self.rain[..., cy:cy+ch, cx:cx+cw] = self.target[..., cy:cy+ch, cx:cx+cw]
            
       
        
        return torch.from_numpy(self.rain.copy()), torch.from_numpy(self.target.copy())

    def __len__(self):        
        return len(self.rains)


class Dataset_real(data.Dataset):
    def __init__(self, file_path='/data/derain_new/Practical_by_Yang', split = 'test', crop = False ,crop_size = 0):
        super(Dataset_real, self).__init__()
        self.path = file_path
        self.all_name = os.listdir(file_path)
        self.rains = [i for i in self.all_name if True]
        self.rains = natsorted(self.rains)
        self.crop_size = crop_size
        self.crop = crop
        self.split = split

    def __getitem__(self, index):

        if self.split == 'test':
            
            self.data   = cv2.imread(os.path.join(self.path,self.rains[index]))[:,:,::-1]
            h,w,c       = self.data.shape
            self.data   = self.data[:,:,:]
            self.data   = self.data.transpose((2,0,1)).astype(np.float32)/255.0

  
        else:
            target_path = os.path.join(self.path,self.rains[index].replace('rainy_image','groundtruth').split('_')[0]+'.jpg').replace('groundtruth','ground_truth')
            
            self.data   = cv2.imread(os.path.join(self.path,self.rains[index]))[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            self.target = cv2.imread(target_path)[:,:,::-1].transpose((2,0,1)).astype(np.float32)/255.0
            if self.crop:
                
                c, h, w = self.data.shape
                hs = np.random.randint(h-self.crop_size)
                ws = np.random.randint(w-self.crop_size)
                self.data   = self.data[:  , hs:hs+self.crop_size, ws:ws+self.crop_size]
                self.target = self.target[:, hs:hs+self.crop_size, ws:ws+self.crop_size]
            else:
                pass
            
        return torch.from_numpy(self.data.copy()) , None

    def __len__(self):        
        return len(self.rains)
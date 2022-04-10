import os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time, math
import numpy as np
import h5py
from torchsummary import summary
from torch_ssim import SSIM
import shutil


import sys
args = sys.argv
print(args)
Net_name = args[1]
module = __import__(Net_name)
Net    = module.Net

config = args[2]
config = __import__(config)
# import config_small as config 



# Training settings
GRAD_CLIP = 0.0  #  1.0

# ===============================Settings============================================
print("===> Loading model")
model = Net()
version = config.train_dataset+'_'+model.name
if config.CUT_OUT:
    version += '_CUT_OUT'
if config.FLIP:
    version += '_flip'
if GRAD_CLIP:
    version += '_gradclip'
print("===> Model name: {}".format(version))
SAVE_EPOCHS = config.SAVE_EPOCHS #[110,120,130,140,150,200] 
# ===================================================================================

def main():
    global model, model_folder
    model_folder = os.path.join(config.checkpoint, version)    
    
    for i in range(100):
        if not os.path.exists(model_folder + "_{}".format(i)):
            model_folder = model_folder + "_{}".format(i)
            os.makedirs(model_folder)
            break
    print("Save folder path: ", model_folder)
    config_file = os.path.basename(config.__file__).split(".")[0] + '.py'
    directory = model_folder
    shutil.copy(config_file, os.path.join(directory, config_file))

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    cuda = config.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = 486
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True
    
    
    print("===> Loading datasets")
    
    train_set = config.train_set
    training_data_loader = DataLoader(dataset=train_set, num_workers=config.threads, batch_size=config.batchSize,drop_last = True, shuffle=config.shuffle)
    data_val = config.test_set

    criterion = nn.L1Loss()

    print("===> Setting GPU")
    ##
    cuda = True
    config.cuda = True
    ##
    if cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    # summary(model, (3, config.input_size, config.input_size))

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    # optionally copy weights from a checkpoint
    if config.pretrained:
        if os.path.isfile(config.pretrained):
            print("=> loading model '{}'".format(config.pretrained))
            weights = torch.load(config.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(config.pretrained))

    print("===> Setting Optimizer")
    if config.use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    

    print("===> Training")
    best_psnr = 0
    for epoch in range(config.start_epoch, config.nEpochs + 1):
        avg_loss = train(training_data_loader, optimizer, model, criterion, epoch, data_val)
        model.eval()
        psnr = 0

        for i in range(data_val.__len__()):

            val_data, val_label = data_val[i]
            val_data, val_label = val_data.numpy(), val_label.numpy()
        
            val_data = Variable(torch.from_numpy(val_data).float()).view(1, 3, val_data.shape[1], val_data.shape[2])

            if config.cuda:
                val_data = val_data.cuda()

            with torch.no_grad():
                val_att, val_out = model(val_data)              
            val_out = val_out.cpu().data[0].numpy()

            psnr += output_psnr_mse(val_label, val_out)
        psnr = psnr / (i + 1)
        if psnr > best_psnr:
            best_psnr = psnr
            save_checkpoint(model, epoch, 0, psnr, avg_loss)
        if epoch in config.SAVE_EPOCHS:        #[110,120,130,140,150]:
            save_checkpoint(model, epoch, 99999, psnr, avg_loss)


def train(training_data_loader, optimizer, model, criterion, epoch, data_val):
    lr = config.learning_rate_scheduling(epoch - 1, config.lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    model.train()

    best_psnr = 0
    loss_sum = 0
    base_loss_sum = 0
    hdr_loss_sum = 0
    att_loss_sum = 0
    ssim_loss_sum = 0
    min_avg_loss = 1
    st = time.time()
    train_psnr = 0
    
    for iteration, batch in enumerate(training_data_loader, 1):

        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        [num_bat, num_c, patch_h, patch_w] = input.shape     

        if config.cuda:
            input = input.cuda()
            label = label.cuda()



        att, out = model(input)
        att_label = label - input
        att_label = att_label.cuda()


         # ========================================save img + train_psnr========================================
        for nob in range(num_bat):
            psnr_label = label[nob].detach().cpu().numpy()[:,:,::-1].transpose((1,2,0))
            psnr_output = out[nob].detach().cpu().numpy()[:,:,::-1].transpose((1,2,0))
            train_psnr += output_psnr_mse(psnr_label, psnr_output)
        #     save_img[nob] = cv2.hconcat([save_img[nob], psnr_output*255.0])
        #     cv2.imwrite('/home/isno/deraining/DHDN/data/test/test{}_{}.jpg'.format(iteration, nob), save_img[nob])
        # ========================================================================================

        base_loss = criterion(out, label)     
        
        if True:
            base_loss = base_loss* 10
        
        
        loss = base_loss


        

        optimizer.zero_grad()
       

        loss.backward()

        if GRAD_CLIP:
            total_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if total_norm> GRAD_CLIP*1000:
                print('total_norm',total_norm)


        optimizer.step()

        loss_sum += loss.item()

        base_loss_sum += base_loss.item()




        if iteration % config.report_step == 0:
            report = "===> Epoch[{}]({}/{}): Train_Loss: {:.6f}, Base_Loss: {:.6f}".format(epoch, iteration, len(training_data_loader), loss_sum / iteration, base_loss_sum / iteration)

            report += ", PSNR: {:.4f} - {}".format(train_psnr/(iteration*num_bat), version)
            print(report)

        if iteration % config.eval_step == 0:

            model.eval()
            psnr = 0

            for i in range(data_val.__len__()):

                val_data, val_label = data_val[i]
                val_data, val_label = val_data.numpy(), val_label.numpy()
                val_data = Variable(torch.from_numpy(val_data).float()).view(1, 3, val_data.shape[1], val_data.shape[2])

                if config.cuda:
                    val_data = val_data.cuda()

                with torch.no_grad():
                    val_att, val_out = model(val_data)
                    
                val_out = val_out.cpu().data[0].numpy()

                psnr += output_psnr_mse(val_label, val_out)

            psnr = psnr / (i + 1)
            avg_loss = loss_sum / iteration
            print("===> Epoch[{}]({}/{}): Train_Loss: {:.10f} Val_PSNR: {:.4f}".format(epoch, iteration,
                                                                                       len(training_data_loader),
                                                                                       avg_loss, psnr))
            model.train()

            if psnr > best_psnr or min_avg_loss > avg_loss:
                if psnr > best_psnr:
                    best_psnr = psnr
                if min_avg_loss > avg_loss:
                    min_avg_loss = avg_loss
                save_checkpoint(model, epoch, iteration, psnr, avg_loss)

    print("training_time: ", time.time() - st)
    avg_loss = loss_sum / len(training_data_loader)
    return avg_loss


def save_checkpoint(model, epoch, iteration, psnr, loss):
    model_out_path = os.path.join(model_folder,"{}_{}_epoch_{}_iter_{}_PSNR_{:.4f}.pth".format(config.test_dataset.upper(),version, epoch, iteration, psnr))
    
    state = {"epoch": epoch, "model": model}
    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


if __name__ == "__main__":
    main()

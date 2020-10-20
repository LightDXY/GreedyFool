import time
from options import BaseOptions
import os
import sys
from data_loader import McDataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torchvision.models as models
import numpy as np
import random
from torchvision.transforms import ToPILImage 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from shutil import copyfile
import generators
import inception_v3
cudnn.benchmark = True 

pool_kernel = 3
Avg_pool = nn.AvgPool2d(pool_kernel, stride=1, padding = int(pool_kernel/2))

def main():
    opt = BaseOptions().parse()
    print (torch.cuda.device_count())
    
    
    ##### Target model loading
    #netT = models.inception_v3(pretrained=True)
    netT = inception_v3.inception_v3(pretrained=False)
    netT.load_state_dict(torch.load('./pretrain/inception_v3_google-1a9a5a14.pth'))
    netT.eval()
    netT.cuda()
    
    ##### Generator loading for distortion map
    netG = generators.Res_ResnetGenerator(3, 1, 16, norm_type='batch', act_type='relu')
    netG = torch.nn.DataParallel(netG, device_ids=range(torch.cuda.device_count()))
    netG.load_state_dict(torch.load('./pretrain/G_imagenet.pth'))
    netG.cuda()
    netG.eval()
    
    
    mean_arr = (0.5,0.5,0.5)
    stddev_arr = (0.5,0.5,0.5)
    im_size = 299
    test_dataset = McDataset(
        opt.dataroot,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_arr, stddev_arr)
        ]),
    )
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,shuffle=True)
    
    file_root = os.path.join('./test/file/',opt.phase,opt.name)
    if not os.path.exists(file_root):
        os.makedirs(file_root)
    filename = sys.argv[0].split('.')[0]
    
    copyfile(filename+'.py',os.path.join(file_root,filename+'.py'))
    copyfile(filename+'.sh',os.path.join(file_root,filename+'.sh'))
    
    root = os.path.join('./test/out/',opt.phase,opt.name)
    eps = opt.max_epsilon * 2 / 255.
    Iter = int(opt.iter)
    print ("Iter {0}".format(Iter))
    print ("EPS {0}".format(opt.max_epsilon))
    
    Baccu = []
    for i in range(1):
        temp_accu = AverageMeter()
        Baccu.append(temp_accu)
    
    num_count = []
    time_count = []
    if opt.max_epsilon >= 128:
        boost = False
    else:
        boost = True
    print ("Boost:{0}".format(boost))
    for idx, data in enumerate(test_loader):
        netG.eval()
        iter_start_time = time.time()
        input_A = data['A']
        input_A = input_A.cuda(async=True)
        real_A = Variable(input_A,requires_grad = False)
        image_names = data['name']
        
        image_hill = netG(real_A * 0.5 + 0.5) * 0.5 + 0.5
        pre_hill = 1- image_hill
        pre_hill = pre_hill.view(1,1, -1)
       
        np_hill = pre_hill.detach().cpu().numpy()
        percen = np.percentile(np_hill, 30)
        pre_hill = torch.max(pre_hill - percen, torch.zeros(pre_hill.size()).cuda())
        np_hill = pre_hill.detach().cpu().numpy()
        percen = np.percentile(np_hill, 75)
        pre_hill /= percen
        pre_hill = torch.clamp(pre_hill, 0, 1)
        pre_hill = Avg_pool(pre_hill) 
        SIZE = int(im_size * im_size)
        
        loss_adv = CWLoss
        
        logist_B = netT(real_A)
        _,target=torch.max(logist_B,1)
        adv = real_A
        ini_num = 1
        grad_num = ini_num
        mask = torch.zeros(1,3,SIZE).cuda()

        temp_eps = eps / 2

        ##### Increasing
        for iters in range(Iter):
            #print (iters)
            temp_A = Variable(adv.data, requires_grad=True)
            logist_B = netT(temp_A)
            _,pre=torch.max(logist_B,1)
            
            if target.cpu().data.float() != pre.cpu().data.float():
                break
            Loss = loss_adv(logist_B, target, -100,False) / real_A.size(0)
            
            netT.zero_grad()
            if temp_A.grad is not None:
                temp_A.grad.data.fill_(0)
            Loss.backward()
            
            grad = temp_A.grad
            abs_grad = torch.abs(grad).view(1,3,-1).mean(1,keepdim = True)
            abs_grad = abs_grad * pre_hill
            if not boost:
                abs_grad = abs_grad * (1 - mask)
            _, grad_sort_idx = torch.sort(abs_grad)
            grad_sort_idx = grad_sort_idx.view( -1)
            grad_idx = grad_sort_idx[-grad_num:]
            mask[0,:,grad_idx] = 1.
            temp_mask = mask.view(1,3,im_size,im_size)
            grad = temp_mask * grad
            
            abs_grad = torch.abs(grad)
            abs_grad = abs_grad / torch.max(abs_grad)
            normalized_grad = abs_grad * grad.sign()
            scaled_grad = normalized_grad.mul(temp_eps)
            temp_A = temp_A - scaled_grad
            temp_A = clip(temp_A, real_A, eps)
            adv = torch.clamp(temp_A, -1, 1)
            if boost:
                grad_num += ini_num
            
        final_adv = adv
        adv_noise = real_A - final_adv
        adv = final_adv
        
        abs_noise = torch.abs(adv_noise).view(1,3,-1).mean(1,keepdim = True)
        temp_mask = abs_noise != 0
        modi_num = torch.sum(temp_mask).data.clone().item()
        
        reduce_num = modi_num
        reduce_count = 0
        ###### Redicing
        if modi_num > 2:
            reduce_idx = 0
            while reduce_idx< reduce_num and reduce_count < 3000:
                reduce_count += 1
                adv_noise = real_A - adv
                
                abs_noise = torch.abs(adv_noise).view(1,3,-1).mean(1,keepdim = True)
                reduce_mask = abs_noise != 0
                reduce_mask = reduce_mask.repeat(1,3,1).float()
                abs_noise[abs_noise == 0] = 3.
                
                reduce_num = torch.sum(reduce_mask).data.clone().item() / 3
                if reduce_num == 1:
                    break
                
                noise_show, noise_sort_idx = torch.sort(abs_noise)
                noise_sort_idx = noise_sort_idx.view( -1)
                
                noise_idx = noise_sort_idx[reduce_idx]
                reduce_mask[0,:,noise_idx] = 0.
                temp_mask = reduce_mask.view(1,3,int(im_size),int(im_size))
                noise = temp_mask * adv_noise
                
                abs_noise = torch.abs(noise)
                abs_noise = abs_noise / torch.max(abs_noise)
                normalized_grad = abs_noise * noise.sign()
                
                with torch.no_grad():
                    netT.eval()
                    step = int(max(int(opt.max_epsilon/10.),1))
                    a = [i for i in range(0, int(opt.max_epsilon+step), step)]
                    search_num = len(a)
                    a = np.asarray(a)*2/255. 
                    ex_temp_eps = torch.from_numpy(a).view(-1,1,1,1).float().cuda()
                    ex_normalized_grad = normalized_grad.repeat(int(search_num),1,1,1)
                    ex_scaled_grad = ex_normalized_grad.mul(ex_temp_eps)
                    ex_real_A = real_A.repeat(int(search_num),1,1,1)
                    ex_temp_A = ex_real_A - ex_scaled_grad
                    ex_temp_A = clip(ex_temp_A, ex_real_A, eps)
                    ex_adv = torch.clamp(ex_temp_A, -1, 1)
                    ex_temp_A = Variable(ex_adv.data, requires_grad=False)
                    ex_logist_B = netT(ex_temp_A)
                    _,pre=torch.max(ex_logist_B,1)
                    comp = torch.eq(target.cpu().data.float(), pre.cpu().data.float())
                    top1 = torch.sum(comp).float() / pre.size(0)
                    if top1 != 1: ##### exists at least one adversarial sample
                        found = False
                        for i in range(int(search_num)):
                            if comp[i] == 0:
                                temp_adv = ex_temp_A[i:i+1]
                                logist_B = netT(temp_adv)
                                _,pre=torch.max(logist_B,1)
                                new_comp = torch.eq(target.cpu().data.float(), pre.cpu().data.float())
                                if torch.sum(new_comp) != 0:
                                    continue
                                found = True
                                adv = temp_adv
                                break
                        if found == False:
                            reduce_idx += 1
                    else:
                        reduce_idx += 1
                        
        
        adv_noise = real_A - adv
        abs_noise = torch.abs(adv_noise).view(1,3,-1).mean(1,keepdim = True)
        temp_mask = abs_noise != 0
        
        reduce_num = torch.sum(temp_mask).data.clone().item()
        L1_X_show = torch.max(torch.abs(real_A - adv)) * 255. / 2

        num_count.append(reduce_num)
      
        logist_B = netT(adv)
        _,pre=torch.max(logist_B,1)
        top1 = torch.sum(torch.eq(target.cpu().data.float(),pre.cpu().data.float()).float()) / input_A.size(0)

        top1 = torch.from_numpy(np.asarray( [(1 - top1)*100 ])).float().cuda(async=True)
        Baccu[0].update(top1[0], input_A.size(0))
        
        time_count.append(time.time() - iter_start_time)
        print('[{it:.2f}][{name}] '
                      'BTOP1: {BTOP1.avg:.2f} '
                      'lossX: {ori:d}/{redu:d} '
                      'Time: {ti:.3f} '
                      'L1: {l1:.1f} '
                      'M&m {mean:.2f}/{median:.2f} '
                      'T&t {tmean:.2f}/{tmedian:.2f} '
                      'Num: {num}'.format(
                          it = float(idx*100)/len(test_loader),
                          name = image_names[0].split('_')[-1],
                          BTOP1 = Baccu[0],
                          ori = int(modi_num),
                          redu = int(reduce_num),
                          l1 = L1_X_show.data.clone().item(),
                          num = grad_num,
                          mean = np.mean(num_count),
                          median = np.median(num_count),
                          tmean = np.mean(time_count),
                          tmedian = np.median(time_count),
                          ti = time.time() - iter_start_time))

       
        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(os.path.join(root,'clean')):
            os.makedirs(os.path.join(root,'clean'))
        if not os.path.exists(os.path.join(root,'adv')):
            os.makedirs(os.path.join(root,'adv'))
        if not os.path.exists(os.path.join(root,'show')):
            os.makedirs(os.path.join(root,'show'))

        hill_imgs = pre_hill.view(pre_hill.size(0),1,im_size,im_size).repeat(1,3,1,1)
        
        if modi_num >= 0.:
            for i in range(input_A.size(0)):
                clip_img = ToPILImage()((adv[i].data.cpu()+ 1) / 2) 
                real_img = ToPILImage()((real_A[i].data.cpu()+ 1) / 2)
                adv_path = os.path.join(root,'adv' ,image_names[i] +'_' +str(int(modi_num))+'.png')
                clip_img.save(adv_path)
                real_path = os.path.join(root,'clean', image_names[i] +'_' +str(int(modi_num))+'.png')
                real_img.save(real_path)
                
                if True:
                    hill_img = ToPILImage()(hill_imgs[i].data.cpu())
                    temp_adv = torch.abs(adv_noise[i].data.cpu())
                    temp_adv = temp_adv / torch.max(temp_adv)
                    temp_adv = 1 - temp_adv
                    adv_img = ToPILImage()(temp_adv)

                    temp_hill = image_hill[i].data.cpu()
                    
                    temp_hill = 1 - temp_hill
                    temp_hill = temp_hill.view(1,im_size,im_size).repeat(3,1,1)
                    
                    temp_hill = ToPILImage()(temp_hill)
                    final = Image.fromarray(np.concatenate([temp_hill, hill_img, real_img,adv_img,clip_img],1))
                    final.save( os.path.join(root,'show', image_names[i] +'_' +str(int(modi_num))+'.png'))



def clip(adv_A,real_A,eps):
    g_x=real_A-adv_A
    clip_gx=torch.clamp(g_x, min=-eps, max=eps)
    adv_x=real_A-clip_gx
    return adv_x

    
def CWLoss(logits, target, kappa=0, tar = True):
    target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].cuda())
    
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)
    
    if tar:
        return torch.sum(torch.max(other-real, kappa))
    else :
        return torch.sum(torch.max(real - other, kappa))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

      
if __name__ == '__main__':
    main()

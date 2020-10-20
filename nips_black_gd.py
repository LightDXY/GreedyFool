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
import torchvision
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.backends.cudnn as cudnn
from shutil import copyfile
import generators
cudnn.benchmark = True 

pool_kernel = 3
Avg_pool = nn.AvgPool2d(pool_kernel, stride=1, padding = int(pool_kernel/2))

def main():
    opt = BaseOptions().parse()
    print (torch.cuda.device_count())
    
    ##### Target model loading
    #netT = models.resnet50(pretrained=True)
    netT = models.vgg16(pretrained=True)
    #netT = models.densenet161(pretrained=True)
    
    netB1 = models.resnet50(pretrained=True)
    netB2 = models.densenet161(pretrained=True)
    netB3 = models.vgg16(pretrained=True)
    netT.eval()
    netT.cuda()
    
    netB1.eval()
    netB1.cuda()
    netB2.eval()
    netB2.cuda()
    netB3.eval()
    netB3.cuda()

    netT.eval()
    netT.cuda()
   
    mean_arr = (0.485, 0.456, 0.406)
    stddev_arr = (0.229, 0.224, 0.225)
    def normalized_eval(net, x):
        x_copy = x.clone()
        x_copy = torch.stack([torchvision.transforms.functional.normalize(x_copy[i], mean_arr, stddev_arr)\
            for i in range(1)])
        return net(x_copy)

    im_size = 224
    test_dataset = McDataset(
        opt.dataroot,
        transform=transforms.Compose([
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
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
    confi = opt.confidence
    print ("Iter {0}".format(Iter))
    print ("EPS {0}".format(opt.max_epsilon))
    print ("Confidence {0}".format(confi))
    Baccu = []
    for i in range(3):
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
        iter_start_time = time.time()
        input_A = data['A']
        input_A = input_A.cuda(async=True)
        real_A = Variable(input_A,requires_grad = False)
        image_names = data['name']
       
        SIZE = int(im_size * im_size)
        
        loss_adv = CWLoss
        
        logist_B = normalized_eval(netT, real_A)
        _,target=torch.max(logist_B,1)
        adv = real_A
        ini_num = 1
        grad_num = ini_num
        mask = torch.zeros(1,3,SIZE).cuda()

        temp_eps = eps / 2
        for iters in range(Iter):
            #print (iters)
            temp_A = Variable(adv.data, requires_grad=True)
            logist_B = normalized_eval(netT, temp_A)
            _,pre=torch.max(logist_B,1)
            Loss = loss_adv(logist_B, target, -100, False) / real_A.size(0)
           
            if target.cpu().data.float() != pre.cpu().data.float():
                temp_loss = Loss.data.cpu().numpy().item()
                if temp_loss < -1 * confi:
                    #print (temp_loss)
                    break
            
            netT.zero_grad()
            if temp_A.grad is not None:
                temp_A.grad.data.fill_(0)
            Loss.backward()
            
            grad = temp_A.grad
            abs_grad = torch.abs(grad).view(1,3,-1).mean(1,keepdim = True)
            
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
            
        adv_noise = real_A - adv
        abs_noise = torch.abs(adv_noise).view(1,3,-1).mean(1,keepdim = True)
        temp_mask = abs_noise != 0
        
        reduce_num = torch.sum(temp_mask).data.clone().item()
        L1_X_show = torch.max(torch.abs(real_A - adv)) * 255. / 2

        num_count.append(reduce_num)
      
        for i, netB in enumerate([netB1, netB2, netB3]):
            logist_B = normalized_eval(netB, real_A)
            _,target=torch.max(logist_B,1)
            logist_B = normalized_eval(netB, adv)
            _,pre=torch.max(logist_B,1)
            top1 = torch.sum(torch.eq(target.cpu().data.float(),pre.cpu().data.float()).float()) / input_A.size(0)
            top1 = torch.from_numpy(np.asarray( [(1 - top1)*100 ])).float().cuda(async=True)
            Baccu[i].update(top1[0], input_A.size(0))

        time_count.append(time.time() - iter_start_time)

        print('[{iter:.2f}][{name}] '
                      'BTOP1: {BTOP1.avg:.2f} '
                      'BTOP2: {BTOP2.avg:.2f} '
                      'BTOP3: {BTOP3.avg:.2f} '
                      'M&m {mean:.2f}/{median:.2f} '
                      'T&t {tmean:.2f}/{tmedian:.2f} '
                      'Num: {num}'.format(
                          iter = float(idx*100)/len(test_loader), 
                          name = image_names[0].split('_')[-1], 
                          BTOP1 = Baccu[0],
                          BTOP2 = Baccu[1],
                          BTOP3 = Baccu[2],
                          num = grad_num,
                          mean = np.mean(num_count),
                          median = np.median(num_count),
                          tmean = np.mean(time_count),
                          tmedian = np.median(time_count)))


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

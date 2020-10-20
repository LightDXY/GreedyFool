import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def weights_init(m, act_type='relu'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if act_type == 'selu':
            n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            m.weight.data.normal_(0.0, 1.0 / math.sqrt(n))
        else:
            m.weight.data.normal_(0.0, 0.02)        
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = ((0.5 ** int(epoch >= 2)) *
                    (0.5 ** int(epoch >= 5)) *
                    (0.5 ** int(epoch >= 8)))
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_iters, gamma=0.1
        )
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler
    
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



def define(input_nc, output_nc, ngf, gen_type, norm='instance',
           act='selu', block=9, gpu_ids=[]):
    network = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if gen_type == 'unet':
        network = UnetGenerator(input_nc, output_nc, ngf, norm, act)
        network.cuda(device_id=gpu_ids[1])
    elif gen_type == 'unet-sc':
        network = UnetGeneratorSC(input_nc, output_nc, ngf, norm, act)
        network.cuda(device_id=gpu_ids[1])
    elif gen_type == 'unet-rec':
        network = RecursiveUnetGenerator(input_nc, output_nc, 8, ngf, norm, act, use_dropout=False, gpu_ids=gpu_ids)
    elif gen_type == 'resnet':
        network = ResnetGenerator(input_nc, output_nc, ngf, norm, act, use_dropout=True, n_blocks=block, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [{}] is not recognized'.format(gen_type))

    weights_init(network, act)
    return network

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.num_gpus = len(self.gpulist)

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)
            
        #self.fc1 = nn.Linear(1000, 2048)
        #self.fc2 = nn.Linear(2048, 4900)
        #self.L_upconv1 = BasicConv2d(4, 64, kernel_size=3, stride=1, padding=1)#74
        #self.L_upconv2 = nn.ConvTranspose2d(4 ,2 ,kernel_size=3 ,stride=2)#149
        
        En = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            En += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]# 75 * 75

        if self.num_gpus == 1:
            mult = 2**n_downsampling
            for i in range(n_blocks):
                En += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        Dn=[]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]
        Dn += [nn.ReflectionPad2d(3)]
        #Dn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        Dn += [nn.Conv2d(ngf, output_nc, kernel_size=8, padding=0)]
        Dn += [nn.Tanh()] 

        self.En = nn.Sequential(*En)
        self.Dn = nn.Sequential(*Dn)
        #self.En.cuda(self.gpulist[0])
        #self.Dn.cuda(self.gpulist[0])

    def forward(self, input):
        #input = input.cuda()
        mid = self.En(input)
        #print (mid.size())
        per = self.Dn(mid)
        return per

class Weight_ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(Weight_ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.num_gpus = len(self.gpulist)

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)
        n_downsampling = 2
        #self.fc1 = nn.Linear(1000, 2048)
        #self.fc2 = nn.Linear(2048, 4900)
        #self.L_upconv1 = BasicConv2d(4, 64, kernel_size=3, stride=1, padding=1)#74
        #self.L_upconv2 = nn.ConvTranspose2d(4 ,2 ,kernel_size=3 ,stride=2)#149
        L1 = [nn.Linear(1000, 2048),
                  self.act,
                  nn.Linear(2048, 5625),
                  self.act]
        mult = 2**n_downsampling
        L2 = [nn.Conv2d(1, ngf , kernel_size=3,
                        stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf),
                  self.act,
                  nn.Conv2d(ngf ,ngf * mult, kernel_size=3,
                            stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf * mult),
                  self.act]
        
        En = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        
        for i in range(n_downsampling):
            mult = 2**i
            En += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]# 75 * 75

        if self.num_gpus == 1:
            mult = 2**n_downsampling
            for i in range(n_blocks):
                En += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        Dn=[]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]
        Dn += [nn.ReflectionPad2d(3)]
        #Dn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        Dn += [nn.Conv2d(ngf, output_nc, kernel_size=8, padding=0)]
        Dn += [nn.Tanh()] 

        self.En = nn.Sequential(*En)
        self.Dn = nn.Sequential(*Dn)
        self.L1 = nn.Sequential(*L1)
        self.L2 = nn.Sequential(*L2)
        #self.En.cuda(self.gpulist[0])
        #self.Dn.cuda(self.gpulist[0])

    def forward(self, input,label):
        weight = self.L1(label)
        weight = weight.view(label.size(0),1,75,75)
        weight = self.L2(weight)
        #input = input.cuda()
        mid = self.En(input)
        #x = torch.cat([mid ,weight],1)
        x = mid +weight
        #print (mid.size())
        per = self.Dn(x)
        return per

        
class Res_ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', addition_res=True):
        assert(n_blocks >= 0)
        super(Res_ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)
            
        #self.fc1 = nn.Linear(1000, 2048)
        #self.fc2 = nn.Linear(2048, 4900)
        #self.L_upconv1 = BasicConv2d(4, 64, kernel_size=3, stride=1, padding=1)#74
        #self.L_upconv2 = nn.ConvTranspose2d(4 ,2 ,kernel_size=3 ,stride=2)#149
        
        En = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            if addition_res:
                for j in range(2):
                    En += [ResnetBlock(int(ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
            En += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]# 75 * 75



        mult = 2**n_downsampling

        self.res1 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res3 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res4 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res5 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res6 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        Dn=[]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]
            if addition_res:
                for j in range(2):
                    Dn += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
        Dn += [nn.ReflectionPad2d(3)]
        #Dn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        Dn += [nn.Conv2d(ngf, output_nc, kernel_size=8, padding=0)]
        Dn += [nn.Tanh()] 

        self.En = nn.Sequential(*En)
        self.Dn = nn.Sequential(*Dn)

    def forward(self, input):
        input = input.cuda()
        mid = self.En(input)
        mid = self.res1(mid)
        mid = self.res2(mid)
        mid = self.res3(mid)
        #mid = self.res4(mid)
        #mid = self.res5(mid)
        #mid = self.res6(mid)
        #print (mid.size())
        per = self.Dn(mid)
        return per
        
class Weight_Res_ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', addition_res=True):
        assert(n_blocks >= 0)
        super(Weight_Res_ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)
            
        #self.fc1 = nn.Linear(1000, 2048)
        #self.fc2 = nn.Linear(2048, 4900)
        #self.L_upconv1 = BasicConv2d(4, 64, kernel_size=3, stride=1, padding=1)#74
        #self.L_upconv2 = nn.ConvTranspose2d(4 ,2 ,kernel_size=3 ,stride=2)#149
        
        En = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            if addition_res:
                for j in range(2):
                    En += [ResnetBlock(int(ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
            En += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]# 75 * 75

        L1 = [nn.Linear(1000, 2048),
                  self.act,
                  nn.Linear(2048, 5625),
                  self.act]
        mult = 2**n_downsampling
        L2 = [nn.Conv2d(1, ngf , kernel_size=3,
                        stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf),
                  self.act,
                  nn.Conv2d(ngf ,ngf * mult, kernel_size=3,
                            stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf * mult),
                  self.act]

        mult = 2**n_downsampling

        self.res1 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res3 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res4 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res5 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res6 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        Dn=[]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]
            if addition_res:
                for j in range(2):
                    Dn += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
        Dn += [nn.ReflectionPad2d(3)]
        #Dn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        Dn += [nn.Conv2d(ngf, output_nc, kernel_size=8, padding=0)]
        Dn += [nn.Tanh()] 

        self.En = nn.Sequential(*En)
        self.Dn = nn.Sequential(*Dn)
        self.L1 = nn.Sequential(*L1)
        self.L2 = nn.Sequential(*L2)

    def forward(self, input,label):
        weight = self.L1(label)
        weight = weight.view(label.size(0),1,75,75)
        weight = self.L2(weight)
        input = input.cuda()
        mid = self.En(input)
        mid = self.res1(mid)
        mid = self.res2(mid)
        mid = self.res3(mid)
        mid = self.res4(mid)
        mid = self.res5(mid)
        mid = self.res6(mid)
        
        mid = mid +weight
        #print (mid.size())
        per = self.Dn(mid)
        return per

class EX_Weight_Res_ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', addition_res=True):
        assert(n_blocks >= 0)
        super(EX_Weight_Res_ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)
            
        #self.fc1 = nn.Linear(1000, 2048)
        #self.fc2 = nn.Linear(2048, 4900)
        #self.L_upconv1 = BasicConv2d(4, 64, kernel_size=3, stride=1, padding=1)#74
        #self.L_upconv2 = nn.ConvTranspose2d(4 ,2 ,kernel_size=3 ,stride=2)#149
        
        En = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            if addition_res:
                for j in range(2):
                    En += [ResnetBlock(int(ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
            En += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]# 75 * 75

        L1 = [nn.Linear(1000, 2048),
                  self.act,
                  nn.Linear(2048, 5625),
                  self.act]
        mult = 2**n_downsampling
        L1 += [nn.Conv2d(1, ngf , kernel_size=3,
                        stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf),
                  self.act,
                  nn.Conv2d(ngf ,ngf * mult, kernel_size=3,
                            stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf * mult),
                  self.act]

        mult = 2**n_downsampling

        self.res1 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res3 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res4 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res5 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res6 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        Dn1=[]
        Dn2=[]
        i=0
        if i == 0:
            mult = 2**(n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]
            if addition_res:
                for j in range(2):
                    Dn += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
        L2 = [nn.Linear(1000, 2048),
                  self.act,
                  nn.Linear(2048, 5625),
                  self.act]
        mult = 2**(n_downsampling - i)
        L2 += [nn.Conv2d(1, ngf , kernel_size=3,
                        stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf),
                  self.act,
                  nn.Conv2d(ngf ,int(ngf * mult / 2), kernel_size=3,
                            stride=1, padding=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  self.act]
        Dn2=[]
        i=1
        if i == 1:
            mult = 2**(n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]
            if addition_res:
                for j in range(2):
                    Dn += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
        Dn += [nn.ReflectionPad2d(3)]
        #Dn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        Dn += [nn.Conv2d(ngf, output_nc, kernel_size=8, padding=0)]
        Dn += [nn.Tanh()] 

        self.En = nn.Sequential(*En)
        self.Dn = nn.Sequential(*Dn)
        self.L1 = nn.Sequential(*L1)
        self.L2 = nn.Sequential(*L2)

    def forward(self, input,label):
        weight = self.L1(label)
        weight = weight.view(label.size(0),1,75,75)
        weight = self.L2(weight)
        input = input.cuda()
        mid = self.En(input)
        mid = self.res1(mid)
        mid = self.res2(mid)
        mid = self.res3(mid)
        mid = self.res4(mid)
        mid = self.res5(mid)
        mid = self.res6(mid)
        
        mid = mid +weight
        #print (mid.size())
        per = self.Dn(mid)
        return per
        
class conWeight_ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(conWeight_ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.num_gpus = len(self.gpulist)

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)
        n_downsampling = 2
        #self.fc1 = nn.Linear(1000, 2048)
        #self.fc2 = nn.Linear(2048, 4900)
        #self.L_upconv1 = BasicConv2d(4, 64, kernel_size=3, stride=1, padding=1)#74
        #self.L_upconv2 = nn.ConvTranspose2d(4 ,2 ,kernel_size=3 ,stride=2)#149
        L1 = [nn.Linear(1000, 2048),
                  self.act,
                  nn.Linear(2048, 5625),
                  self.act]
        mult = 2**n_downsampling
        L2 = [nn.Conv2d(1, ngf , kernel_size=3,
                        stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf),
                  self.act,
                  nn.Conv2d(ngf ,ngf * mult, kernel_size=3,
                            stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf * mult),
                  self.act]
        
        En = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        
        for i in range(n_downsampling):
            mult = 2**i
            En += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]# 75 * 75

        if self.num_gpus == 1:
            mult = 2**n_downsampling
            for i in range(n_blocks):
                En += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        mult = 2**n_downsampling
        Dn = [nn.Conv2d(ngf * mult * 2 , ngf * mult, kernel_size=3,
                             stride=1, padding=1, bias=use_bias),
                       norm_layer(ngf * mult ),
                       self.act]# 75 * 75
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]
        Dn += [nn.ReflectionPad2d(3)]
        #Dn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        Dn += [nn.Conv2d(ngf, output_nc, kernel_size=8, padding=0)]
        Dn += [nn.Tanh()] 

        self.En = nn.Sequential(*En)
        self.Dn = nn.Sequential(*Dn)
        self.L1 = nn.Sequential(*L1)
        self.L2 = nn.Sequential(*L2)
        #self.En.cuda(self.gpulist[0])
        #self.Dn.cuda(self.gpulist[0])

    def forward(self, input,label):
        weight = self.L1(label)
        weight = weight.view(label.size(0),1,75,75)
        weight = self.L2(weight)
        #input = input.cuda()
        mid = self.En(input)
        x = torch.cat([mid ,weight],1)
        #print (mid.size())
        per = self.Dn(x)
        #res = torch.clamp(per+input, min=-1.0, max=1.0)
        return per
        
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

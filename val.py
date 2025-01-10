import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
# from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
import function
from conf import settings
# from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from utils_test import evaluation_multi_proj
from utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from models.BEIT3.beit3 import EvfSamModel
from transformers import AutoTokenizer
from models.BEIT3.configuration_evf import EvfConfig


def main():
    args = cfg.parse_args()

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    '''load pretrained model'''
    checkpoint = torch.load(args.weights)['sam']
    net.load_state_dict(checkpoint)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(GPUdevice)
    bn = bn.to(GPUdevice)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(GPUdevice)

    proj_layer = MultiProjectionLayer(base=64).to(GPUdevice)

    config = EvfConfig()
    config.mm_extractor_scale = "base" 
    beit3 = EvfSamModel(config).to(GPUdevice) 
    tokenizer = AutoTokenizer.from_pretrained('/home/interimuser/qsc/SAM-AD-MVTEC/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/')
    
    
    beit3_weight = torch.load('/home/interimuser/qsc/SAM-AD-MVTEC/logs/mvtec_2024_11_05_16_27_01/Model/best_beit3_checkpoint.pth')['beit3']
    beit3.load_state_dict(beit3_weight)
    
    rd = torch.load('/home/interimuser/qsc/Revisiting-Reverse-Distillation-main/RD++_checkpoint_result/111/wres50_111.pth', map_location='cpu')
    proj_layer.load_state_dict(rd['proj'])
    bn.load_state_dict(rd['bn'])
    decoder.load_state_dict(rd['decoder'])

    # args.path_helper = checkpoint['path_helper']
    # logger = create_logger(args.path_helper['log_path'])
    # print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    net.eval()
    bn.eval()
    decoder.eval()
    proj_layer.eval()
    beit3.eval()


    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    train_loader, test_loaders = get_dataloader(args)

    '''begain valuation'''
    best_acc = 0.0
    best_tol = 1e4

    if args.mod == 'sam_adpt':
        for subclass, test_loader in test_loaders.items():
            logger.info(f'Testing on subclass: {subclass}')
            image_auroc, image_ap, pixel_auroc, pixel_ap = function.validation_sam(args, test_loader, 100, net, encoder, proj_layer, bn, decoder, beit3, tokenizer, subclass)
            logger.info(f'Subclass: {subclass} || IMAGE_AUROC: {image_auroc}, IMAGE_AP: {image_ap}, PIXEL_AUROC: {pixel_auroc}, PIXEL_AP: {pixel_ap}')



if __name__ == '__main__':
    main()

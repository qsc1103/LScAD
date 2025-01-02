import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

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

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights, strict=False)

    optimizer_sam = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer_sam, step_size=10, gamma=0.5)

    encoder, bn = resnet34(pretrained=True)
    encoder = encoder.to(GPUdevice)
    bn = bn.to(GPUdevice)
    encoder.eval()

    decoder = de_resnet34(pretrained=False)
    decoder = decoder.to(GPUdevice)

    proj_layer = MultiProjectionLayer(base=16).to(GPUdevice)

    config = EvfConfig()
    config.mm_extractor_scale = "base" 
    beit3 = EvfSamModel(config).to(GPUdevice) 
    tokenizer = AutoTokenizer.from_pretrained('models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/')
    
    
    #beit3_weight = torch.load('/home/interimuser/qsc/SAM-AD-MVTEC/BEIT3_weight/beit3_base_patch16_224.pth')['model']
    #beit3.load_state_dict(beit3_weight)

    bn.train()
    proj_layer.train()
    decoder.train()
    beit3.train()
    net.train()
    #beit3.eval()

    '''load pretrained model'''
    if args.weights != 0:
        checkpoint = torch.load(args.weights)['sam']
        net.load_state_dict(checkpoint)

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, test_loaders = get_dataloader(args)  # test_loaders is now a dictionary

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

    '''begin training'''
    best_avg_img_auroc = 0.0  # Store the best average AUROC
    best_pixel_auroc = 0.0    # Store the best pixel AUROC

    for epoch in range(settings.EPOCH):
        net.train()
        time_start = time.time()

        loss = function.train_sam(args, net, encoder, bn, decoder, proj_layer, beit3, tokenizer, optimizer_sam, nice_train_loader, epoch, writer, vis=args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch % args.val_freq == 10 or epoch == settings.EPOCH:
            total_image_auroc = 0.0  # Sum of AUROCs for calculating the average
            subclass_count = len(test_loaders)  # Number of subclasses

            for subclass, test_loader in test_loaders.items():
                logger.info(f'Testing on subclass: {subclass}')
                image_auroc, image_ap, pixel_auroc, pixel_ap = function.validation_sam(args, test_loader, epoch, net, encoder, proj_layer, bn, decoder, beit3, tokenizer, args.subclass)
                logger.info(f'Subclass: {subclass} || IMAGE_AUROC: {image_auroc}, IMAGE_AP: {image_ap}, PIXEL_AUROC: {pixel_auroc}, PIXEL_AP: {pixel_ap}')

                # Add the AUROC of this subclass to the total
                total_image_auroc += image_auroc

                # Check if current pixel_auroc is the best
                if pixel_auroc > best_pixel_auroc:
                    best_pixel_auroc = pixel_auroc
                    logger.info(f'Best pixel_auroc: {best_pixel_auroc} @ subclass: {subclass}')
                    is_best=True

                    # Save args.net and beit3 model weights
                    torch.save({
                        'sam': net.state_dict()
                    }, os.path.join(args.path_helper['ckpt_path'], 'best_sam_checkpoint.pth'))

                    torch.save({
                        'beit3': beit3.state_dict()
                    }, os.path.join(args.path_helper['ckpt_path'], 'best_beit3_checkpoint.pth'))
                    
                    
                    print('Best SAM model saved with average PIXEL_AUROC:', best_pixel_auroc)
                    print('Best BEIT3 model saved with average PIXEL_AUROC:', best_pixel_auroc)
                    
            # Calculate the average image AUROC over all subclasses
            avg_image_auroc = total_image_auroc / subclass_count
            logger.info(f'Average IMAGE_AUROC: {avg_image_auroc} || @ epoch {epoch}')

            # Save the model if the average image AUROC is better than the best so far
            if avg_image_auroc > best_avg_img_auroc:
                best_avg_img_auroc = avg_image_auroc
                logger.info(f'Best avg_image_auroc: {best_avg_img_auroc} @ epoch {epoch}')

                # Save proj_layer, decoder, and bn when img_auroc is the best
                torch.save({
                    'proj': proj_layer.state_dict(),
                    'decoder': decoder.state_dict(),
                    'bn': bn.state_dict()
                }, os.path.join(args.path_helper['ckpt_path'], 'best_rd_checkpoint.pth'))

                print('Best small model saved with average IMAGE_AUROC:', best_avg_img_auroc)

    writer.close()


if __name__ == '__main__':
    main()


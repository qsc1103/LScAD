import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, auc, roc_curve
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
#from models.discriminatorlayer import discriminator
from conf import settings
from utils import *
from scipy.ndimage import gaussian_filter
import pandas as pd
from skimage import measure
from statistics import mean
from utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from models.BEIT3.beit3 import EvfSamModel
from transformers import AutoTokenizer
from models.BEIT3.configuration_evf import EvfConfig





args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []




def train_sam(args, net: nn.Module, encoder, bn, decoder, proj_layer, beit3, tokenizer, optimizer, train_loader, epoch, writer, schedulers=None, vis=50):
    hard = 0
    epoch_loss = 0
    ind = 0
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    encoder, bn, decoder, proj_layer = encoder, bn, decoder, proj_layer
    beit3, tokenizer = beit3, tokenizer

    encoder.eval()
    bn.train()
    proj_layer.train()
    decoder.train()
    beit3.train()

    proj_loss = Revisit_RDLoss()

    # 优化器
    optimizer_proj = optim.Adam(list(proj_layer.parameters()), lr=0.001, betas=(0.5, 0.999))
    optimizer_distill = optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=0.005, betas=(0.5, 0.999))
    optimizer_proj.zero_grad()
    optimizer_distill.zero_grad()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # 数据加载和预处理
            ori_img = pack['ori_image'].to(dtype=torch.float32, device=GPUdevice)
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            imgs_f = pack['frequency_img'].to(dtype=torch.float32, device=GPUdevice)
            img_vis = pack['img_vis'].to(dtype=torch.float32, device=GPUdevice)

            pt = pack['pt']
            point_labels = pack['p_label']
            pt_box = pack['pt_box']
            name = pack['image_meta_dict']['filename_or_obj']
            showp = pt
            prompt_text = pack['text_prompt']

            # 使用 tokenizer 编码 prompt
            encoded_input = tokenizer(prompt_text, return_tensors='pt').to(GPUdevice)
            input_ids = encoded_input['input_ids']
            attention_masks = encoded_input['attention_mask']

            # 使用 BEiT-3 获取文本嵌入
            images_evf = imgs  # 假设 imgs_f 是与图像对应的特征
            beit3_output = beit3(
                images_evf=images_evf,
                input_ids=input_ids,
                attention_masks=attention_masks
            )
            feat = beit3_output  # 获取特征
            batch_size = feat.size(0)
            pred_masks = []

            # 将 feat 用于 prompt_encoder
            # 在后续代码中，需要将 feat 传递给 prompt_encoder

            # 编码器和解码器部分
            inputs = encoder(ori_img)
            inputs_noise = encoder(imgs)
            (feature_space_noise, feature_space) = proj_layer(inputs, features_noise=inputs_noise)
            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)
            outputs = decoder(bn(feature_space))
            L_distill = loss_fucntion(inputs, outputs)
            loss_distill = L_distill + 0.2 * L_proj

            ind += 1
            b_size, c, w, h = imgs.size()

            if point_labels.clone().flatten()[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if len(point_labels.shape) == 1:  # 只有一个点提示
                    coords_torch = coords_torch[None, :, :]
                    labels_torch = labels_torch[None, :]
                    showp = showp[None, :, :]
                pt = (coords_torch, labels_torch)

            '''模型前向传播'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True

            imge = net.image_encoder(imgs, imgs_f)








            (
                sparse_embeddings,
                dense_embeddings,
            ) = net.prompt_encoder(
                points=pt,
                boxes=None,
                masks=None,
                text_embeds=None,
            )
            
            sparse_embeddings = sparse_embeddings.to(feat.dtype)

            if args.net == 'sam':
                pred_point, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=(args.multimask_output > 1),)
                    
                    
                    
            pred_point = F.interpolate(pred_point, size=(args.out_size, args.out_size))


            (
                sparse_embeddings,
                dense_embeddings,
            ) = net.prompt_encoder(
                points=None,
                boxes=pt_box,
                masks=None,
                text_embeds=None,
            )
            
            sparse_embeddings = sparse_embeddings.to(feat.dtype)

            if args.net == 'sam':
                pred_box, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=(args.multimask_output > 1),)
                    
            pred_box = F.interpolate(pred_box, size=(args.out_size, args.out_size))












            # 使用 BEiT-3 的特征作为文本嵌入，传递给 prompt_encoder
            for i in range(batch_size):
                feat_i = feat[i].unsqueeze(0)  # 获取当前样本的特征

                # 将 feat_i 用于 prompt_encoder
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = net.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=feat_i,
                )

                sparse_embeddings = sparse_embeddings.to(feat_i.dtype)

                # 根据模型类型处理不同的输入
                if args.net == 'sam':
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge[i].unsqueeze(0),  # 当前样本的图像特征
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=(args.multimask_output > 1),
                    )
                elif args.net == 'mobile_sam':
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge[i].unsqueeze(0),
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                elif args.net == "efficient_sam":
                    sparse_embeddings = sparse_embeddings.view(
                        sparse_embeddings.shape[0],
                        1,
                        sparse_embeddings.shape[1],
                        sparse_embeddings.shape[2],
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge[i].unsqueeze(0),
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        multimask_output=False,
                    )

                # 保存预测的掩码
                pred_masks.append(pred)

            # 将所有预测的掩码合并回 batch 维度
            pred_masks = torch.cat(pred_masks, dim=0)

            # 调整预测结果尺寸
            pred_text = F.interpolate(pred_masks, size=(args.out_size, args.out_size))
            
            pred_com = (pred_box+pred_box+pred_text)/3

            loss = lossfunc(pred_com, masks)
            loss_total = loss_distill + loss

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss_total.backward()

            optimizer_proj.step()
            optimizer_proj.zero_grad()

            optimizer_distill.step()
            optimizer_distill.zero_grad()

            optimizer.step()
            optimizer.zero_grad()

            '''可视化部分'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name[:2]:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image_0(img_vis, pred_point, pred_box, pred_text, masks,
                                os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'),
                                reverse=False, points=showp, boxes=pt_box, p_label=point_labels)
            pbar.update()
    return loss







def validation_sam(args, val_loader, epoch, net: nn.Module, encoder, proj, bn, decoder, beit3, tokenizer, subclass):
    # 设置模型为评估模式
    net.eval()
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    beit3.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # 验证集中的批次数量
    all_image_scores_RD = []  # 用于存储图像分数
    all_true_labels = []  # 用于存储所有真实标签
    all_preds = []  # 用于存储所有预测掩码
    all_masks = []  # 用于存储所有真实掩码
    all_binary_masks = []  # 用于存储所有二值化掩码
    subclass = subclass

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # 第一次遍历验证集，用于计算最优阈值
    with tqdm(total=n_val, desc='Validation (Finding Best Threshold)', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            img = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            mask = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            true_labels = pack["has_anomaly"].cpu().numpy()  # 获取异常检测的真实标签

            # 编码器和解码器前向传播
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # 计算图像级别分数
            image_scores_RD = np.max(anomaly_map)  # 计算每个图像的分数
            all_image_scores_RD.append(image_scores_RD)  # 将每个图像的分数加入列表
            all_true_labels.extend(true_labels)

            pbar.update()

    # 使用 ROC 曲线计算最佳阈值
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_image_scores_RD)
    gmeans = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmeans)
    best_threshold = thresholds[best_idx]

    print(f"Best threshold determined from validation: {best_threshold}")

    # 第二次遍历验证集，使用最优阈值生成 prompt，并进行分割和二值化掩码生成
    with tqdm(total=n_val, desc='Validation (Generating Predictions)', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            img = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            mask = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            img_f = pack['frequency_img'].to(dtype=torch.float32, device=GPUdevice)
            img_vis = pack['img_vis'].to(dtype=torch.float32, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            pt = pack['pt']
            point_labels = pack['p_label']
            pt_box = pack['pt_box']
            
            showp = pt
            

            # 编码器和解码器前向传播
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # 使用图像级别分数确定 prompt
            image_scores_RD = np.max(anomaly_map)
            if image_scores_RD > best_threshold:
                prompt_text = "defect"
            else:
                prompt_text = "good"

            # 固定阈值生成二值化掩码 (不使用最佳阈值)
            fixed_threshold = 0.5  # 这里是你希望的固定阈值
            
            prompt_text = pack['text_prompt']
            
            binary_mask = (anomaly_map > best_threshold).astype(np.uint8)  # 使用固定阈值生成二值化掩码
            binary_mask = np.squeeze(binary_mask)
            binary_mask_tensor = torch.from_numpy(binary_mask).to(device=GPUdevice)
            binary_mask_tensor_4d = binary_mask_tensor.unsqueeze(0).unsqueeze(0).float()
            binary_mask_tensor_4d = F.interpolate(binary_mask_tensor_4d, size=(args.out_size, args.out_size), mode='bilinear', align_corners=False)
            
            
            if torch.any(binary_mask_tensor):  # If there is any non-zero value in the mask, anomaly detected
                point_label, points = random_click(binary_mask)  # Generate click points based on binarized mask
                box = find_largest_bounding_box(binary_mask)  # Find the largest bounding box
                pt_box = box
            else:  # If the mask is all black, no anomaly detected
                point_label, points = random_click_black(binary_mask)  # Generate random click points on black mask
                pt_box = np.array([0, 0, 0, 0])
            
            
            pt = points
                
                
            if point_labels.clone().flatten()[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_label, dtype=torch.int, device=GPUdevice)
                if len(point_label.shape) == 1:  # If a single point
                    coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                pt = (coords_torch, labels_torch)
            
            
            pt_box = [pt_box]
            pt_box = torch.tensor(pt_box, dtype=torch.float32, device=GPUdevice)
    
            
            

            # 使用 tokenizer 编码文本提示
            encoded_input = tokenizer(prompt_text, return_tensors='pt').to(GPUdevice)
            input_ids = encoded_input['input_ids']
            attention_masks = encoded_input['attention_mask']

            # 使用 BEiT-3 获取文本嵌入
            images_evf = img  # 假设 img 是与图像对应的特征
            beit3_output = beit3(
                images_evf=images_evf,
                input_ids=input_ids,
                attention_masks=attention_masks
            )
            feat = beit3_output  # 获取文本嵌入特征

            # 使用文本嵌入进行掩码预测
            with torch.no_grad():
                imge = net.image_encoder(img, img_f)
                
                if args.net == 'sam': 
                    
                    sparse_embeddings_point, dense_embeddings_point = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                        text_embeds=None  # 仅使用文本嵌入
                    )

                    pred_point, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings_point,
                        dense_prompt_embeddings=dense_embeddings_point,
                        multimask_output=(args.multimask_output > 1)
                    )
                
                # 调整预测掩码的大小
                pred = F.interpolate(pred_point, size=(args.out_size, args.out_size))

                # 存储预测的掩码、真实掩码和二值化掩码
                all_preds.append(pred)
                all_masks.append(mask)
                all_binary_masks.append(binary_mask_tensor_4d)

                # 可视化中间步骤
                if ind % args.vis == 0:
                    for na in name[:2]:
                        img_name = na.split('/')[-1].split('.')[0]
                        namecat = img_name + '+'
                    vis_image(img_vis, pred_point, mask, anomaly_map, binary_mask, os.path.join(args.path_helper['sample_path'], subclass + namecat + 'epoch+' + str(epoch) + '.jpg'), points = pt, p_label = point_label)

            pbar.update()

    # 将预测和掩码转换为 numpy 数组以进行 AP 和 AUROC 计算
    all_pred_flat = np.concatenate([pred.flatten().cpu().numpy() for pred in all_preds])
    all_mask_flat = np.concatenate([mask.flatten().cpu().numpy() for mask in all_masks])

    # 计算像素级别的平均精度 (AP) 和 AUROC
    pixel_ap = average_precision_score(all_mask_flat, all_pred_flat)
    pixel_auroc = eval_seg_auroc(torch.cat(all_preds), torch.cat(all_masks))

    # 计算图像级别的 AUROC 和 AP
    image_auroc = roc_auc_score(all_true_labels, all_image_scores_RD)
    image_ap = average_precision_score(all_true_labels, all_image_scores_RD)

    return image_auroc, image_ap, pixel_auroc, pixel_ap












warnings.filterwarnings('ignore')
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap
















def validation_sam_0(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0, 0, 0, 0), (0,) * args.multimask_output * 2
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    all_preds = []
    all_masks = []
    all_image_scores = []
    all_true_labels = []

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            true_labels = pack["has_anomaly"].cpu().numpy()  # 获取真实标签
            imgs_f = pack['frequency_img'].to(dtype = torch.float32, device = GPUdevice)
            img_vis = pack['img_vis'].to(dtype=torch.float32, device=GPUdevice)

            if 'pt' not in pack or args.thd:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']

            pt_box = pack['pt_box']
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:, :, buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[..., buoy: buoy + evl_ch]
                masks = masksw[..., buoy: buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size, c, w, h = imgs.size()
                longsize = w if w >= h else h

                if point_labels.clone().flatten()[0] != -1:
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if len(point_labels.shape) == 1:  # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                    pt = (coords_torch, labels_torch)

                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()

                imgs = imgs.to(dtype=mask_type, device=GPUdevice)

                # Forward pass
                with torch.no_grad():
                    imge = net.image_encoder(imgs, imgs_f)
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=(args.multimask_output > 1),
                        )
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(
                            se.shape[0],
                            1,
                            se.shape[1],
                            se.shape[2],
                        )
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            multimask_output=False,
                        )

                    # Resize to the ordered output size
                    pred = F.interpolate(pred, size=(args.out_size, args.out_size))
                    tot += lossfunc(pred, masks)

                    # 保存所有预测和真实掩码以便之后计算AUROC
                    all_preds.append(pred)
                    all_masks.append(masks)

                    # 计算图像级别异常分数
                    out_mask_averaged = F.avg_pool2d(pred[:, 0, :, :], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
                    image_scores = np.max(out_mask_averaged, axis=(1, 2))  # [b]

                    # 保存图像级别异常分数和标签
                    all_image_scores.extend(image_scores)
                    all_true_labels.extend(true_labels)

                    # Visualization
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name[:2]:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        vis_image(img_vis, pred, masks, os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'),
                                  reverse=False, points=showp, boxes=pt_box, p_label=point_labels)

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    # 计算总体的Pixel AUROC
    pixel_auroc = eval_seg_auroc(torch.cat(all_preds), torch.cat(all_masks))

    # 计算图像级别AUROC
    image_auroc = roc_auc_score(all_true_labels, all_image_scores)
    all_true_labels_flat = [label.item() for label in all_true_labels]
    #print(f'all_true_labels:{all_true_labels_flat}')
    #print(f'all_image_scores:{all_image_scores}')

    return tot / n_val, tuple([a / n_val for a in mix_res]), image_auroc, pixel_auroc









def compute_pro(masks, amaps, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc












def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )
import torch
from torch.nn import functional as F
import cv2
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.metrics import roc_auc_score, auc
from skimage import measure
from statistics import mean
from scipy.ndimage import gaussian_filter
import warnings



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
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
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



# def evaluation_multi_proj(encoder,proj,bn, decoder, dataloader,device):
#     encoder.eval()
#     proj.eval()
#     bn.eval()
#     decoder.eval()
#     gt_list_px = []
#     pr_list_px = []
#     gt_list_sp = []
#     pr_list_sp = []
#     aupro_list = []
#     with torch.no_grad():
#         for (img, gt, label, _, _) in dataloader:
#
#             img = img.to(device)
#             inputs = encoder(img)
#             features = proj(inputs)
#             outputs = decoder(bn(features))
#             anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
#             anomaly_map = gaussian_filter(anomaly_map, sigma=4)
#             gt[gt > 0.5] = 1
#             gt[gt <= 0.5] = 0
#             if label.item()!=0:
#                 print(gt.shape)
#                 aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
#                                               anomaly_map[np.newaxis,:,:]))
#             gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
#             pr_list_px.extend(anomaly_map.ravel())
#             gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
#             pr_list_sp.append(np.max(anomaly_map))
#
#         auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
#         auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
#     return auroc_px, auroc_sp, round(np.mean(aupro_list),4)







import os
import cv2
import matplotlib.pyplot as plt


def evaluation_multi_proj(encoder, proj, bn, decoder, dataloader, device, save_dir=None):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()

    # 创建保存路径
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for batch_idx, (img, gt, label, _, file_name) in enumerate(dataloader):  # 假设 file_name 包含图像文件名
            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # 获取异常图的最大值和最小值
            anomaly_map_max = anomaly_map.max()
            anomaly_map_min = anomaly_map.min()

            # 保存每个异常图与真实掩码
            if save_dir is not None:
                # 提取 'color\\000' 部分
                path_parts = os.path.split(file_name[0])  # 分割出 ('carpet\\test\\color', '000.png')
                folder_name = os.path.basename(os.path.normpath(path_parts[0]))  # 提取 'color'
                base_name = os.path.splitext(path_parts[1])[0]  # 提取 '000'，去掉扩展名
                save_name = os.path.join(folder_name, base_name)  # 拼接 'color\\000'

                # 构建保存文件路径
                save_path = os.path.join(save_dir, f"{save_name}_anomaly_and_gt.png")

                # 确保保存路径中的目录存在，若不存在则创建
                save_folder = os.path.dirname(save_path)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # 归一化异常图，确保像素值在 [0, 255] 之间
                normalized_anomaly_map = min_max_norm(anomaly_map) * 255
                normalized_anomaly_map = normalized_anomaly_map.astype(np.uint8)

                # 将 gt 和 anomaly_map 转换为 numpy 格式
                gt = gt.cpu().numpy().squeeze()

                # 创建图像组合，使用 Matplotlib
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 创建1行2列的子图
                ax[0].imshow(gt, cmap='gray')
                ax[0].set_title('Ground Truth Mask')
                ax[0].axis('off')

                # 在标题中显示异常图的最大值和最小值
                ax[1].imshow(normalized_anomaly_map, cmap='jet')
                ax[1].set_title(f'Anomaly Map\nMax: {anomaly_map_max:.2f}, Min: {anomaly_map_min:.2f}')
                ax[1].axis('off')

                # 保存组合后的图像
                plt.tight_layout()
                plt.savefig(save_path, dpi=100)
                plt.close()

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if label.item() != 0:
                gt = np.expand_dims(gt, axis=0)
                aupro_list.append(compute_pro(gt.astype(int), anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
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

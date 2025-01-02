import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage import morphology
import cfg
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from skimage import morphology
from dataset.perlin import rand_perlin_2d_np
from utils import random_click, random_box, random_click_black, find_disconnected_bounding_boxes
import matplotlib.patches as patches

args = cfg.parse_args()


class AnomalyGenerator:

    def generate_target_foreground_mask(self, img: np.ndarray, dataset: str, subclass: str) -> np.ndarray:
        # convert RGB into GRAY scale
        if dataset == 'mvtec':
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) * 255
            img_gray = img_gray.astype(np.uint8)

            if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor', 'multi', 'multi_textured']:
                return np.ones_like(img_gray)
            if subclass == 'pill':
                _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                target_foreground_mask = target_foreground_mask.astype(np.bool_).astype(np.int32)
            elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
                _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
                target_foreground_mask = target_foreground_mask.astype(np.bool_).astype(np.int32)
            elif subclass in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
                _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                target_background_mask = target_background_mask.astype(np.bool_).astype(np.int32)
                target_foreground_mask = 1 - target_background_mask
            else:
                raise NotImplementedError("Unsupported foreground segmentation category")

            target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(10))
            target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(10))
            return target_foreground_mask

    def generate_perlin_noise_mask(self, shape=(256, 256), scale=(4, 4)):
        perlin_noise = rand_perlin_2d_np(shape, scale)
        perlin_mask = np.where(perlin_noise > 0.5, 1, 0)
        return perlin_mask


class MVTecTrainDataset(Dataset):

    def __init__(self, args, root_dir, anomaly_source_path, subclass, resize_shape=None, prompt='click'):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')) +
                                  glob.glob(os.path.join(root_dir, '*.jpg')) +
                                  glob.glob(os.path.join(root_dir, '*.bmp')))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*.jpg"))
        self.subclass = subclass

        self.prompt = prompt  # 新增的 prompt 参数，用于控制是否生成点击提示
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.anomaly_generator = AnomalyGenerator()

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]])
        return aug

    def augment_image(self, image, anomaly_source_path, dataset='mvtec', subclass='bottle'):
        aug = self.randAugmenter()
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Generate Perlin noise mask
        perlin_mask = self.anomaly_generator.generate_perlin_noise_mask(self.resize_shape)

        # Generate target foreground mask
        target_foreground_mask = self.anomaly_generator.generate_target_foreground_mask(image, dataset, subclass)

        # Combine Perlin mask and target foreground mask
        combined_mask = perlin_mask * target_foreground_mask

        # # 显示 image, combined_mask 和 augmented_image
        # plt.figure(figsize=(12, 4))
        #
        # plt.subplot(1, 3, 1)
        # plt.imshow(perlin_mask, cmap='gray')
        # plt.title('Perlin Mask')
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(target_foreground_mask, cmap='gray')
        # plt.title('Augmented Image')
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(combined_mask, cmap='gray')
        # plt.title('Combined Mask')
        # plt.axis('off')
        #
        # plt.show()

        img_thr = anomaly_img_augmented.astype(np.float32) * combined_mask[..., np.newaxis] / 255.0
        beta = torch.rand(1).numpy()[0] * 8
        augmented_image = image * (1 - combined_mask[..., np.newaxis]) + \
                          (1 - beta) * img_thr + \
                          beta * image * combined_mask[..., np.newaxis]

        augmented_image = np.clip(augmented_image, 0, 1)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            return image.astype(np.float32), np.zeros_like(combined_mask, dtype=np.float32), np.array([0.0],
                                                                                                      dtype=np.float32), anomaly_source_img
        else:
            augmented_image = augmented_image.astype(np.float32)
            return augmented_image, combined_mask.astype(np.float32), np.array([1.0],
                                                                               dtype=np.float32), anomaly_source_img

    def transform_image(self, image_path, anomaly_source_path, subclass):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        add_anomaly = torch.rand(1).numpy()[0]

        if add_anomaly > 0:
            augmented_image, anomaly_mask, has_anomaly, anomaly_source_img = self.augment_image(image,
                                                                                                anomaly_source_path,
                                                                                                'mvtec', subclass)
        else:
            augmented_image = image.copy()
            anomaly_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)  # 生成二维掩码
            has_anomaly = np.array([0.0], dtype=np.float32)

        # 确保 anomaly_mask 是三维的
        if anomaly_mask.ndim == 2:
            anomaly_mask = np.expand_dims(anomaly_mask, axis=2)

        augmented_image255 = augmented_image * 255.0
        imagegray = cv2.cvtColor(augmented_image255, cv2.COLOR_BGR2GRAY)
        # imagegray = cv2.equalizeHist(imagegray.astype(np.uint8))

        # fft
        f = np.fft.fft2(imagegray)
        fshift = np.fft.fftshift(f)

        # BHPF
        rows, cols = imagegray.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        d = 30  # cutoff frequency
        n = 4  # BHPF order
        epsilon = 1e-6  # avoid dividing zero
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
        maska = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
        fshift_filtered = fshift * maska

        # inverse fft
        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        imagegray = np.real(image_filtered).astype(np.float32)
        imagegray = imagegray[:, :, None]

        # print(imagegray)

        image = np.transpose(image, (2, 0, 1))
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        imagegray = np.transpose(imagegray, (2, 0, 1))

        return image, augmented_image, anomaly_mask, has_anomaly, anomaly_source_img, imagegray

    def __getitem__(self, idx):
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image_path = self.image_paths[idx]  # 获取图像路径
        image, augmented_image, anomaly_mask, has_anomaly, anomaly_source_img, frequency_img = self.transform_image(
            image_path, self.anomaly_source_paths[anomaly_source_idx], subclass=self.subclass)

        point_label = 1  # 默认的点标签
        num_points = 10

        if self.prompt == 'click' and has_anomaly == 1.0:
            # 如果启用点提示，并且有异常，生成点击提示点
            point_label, pt = random_click(anomaly_mask, point_label, num_points)
            pt = pt[:, 1:]
        else:
            point_label = 0
            point_label, pt = random_click_black(anomaly_mask, point_label, num_points)

        box_num, boxes = find_disconnected_bounding_boxes(anomaly_mask)

        image = torch.tensor(image)
        augmented_image = torch.tensor(augmented_image)
        anomaly_mask = torch.tensor(anomaly_mask)
        frequency_img = torch.tensor(frequency_img)

        frequency_img = frequency_img / 128.0

        img_vis = augmented_image

        transform_train = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_train_seg = transforms.Compose([
            transforms.Resize((args.out_size, args.out_size))
        ])

        state = torch.get_rng_state()
        ori_image = transform_train(image)
        augmented_image = transform_train(augmented_image)

        torch.set_rng_state(state)

        anomaly_mask = transform_train_seg(anomaly_mask).int()

        # 使用 os.path.basename 提取图像的文件名
        name = os.path.basename(image_path)
        name = f"{self.subclass}_{name}"
        image_meta_dict = {'filename_or_obj': name}

        # print(f"Frequency Image - Max: {frequency_img.max()}, Min: {frequency_img.min()}")
        # print(f"Image - Max: {image.max()}, Min: {image.min()}")
        text_prompt = 'defect' if has_anomaly[0] == 1 else 'good'

        sample = {
            'ori_image': image,
            'img_vis': img_vis,
            'label': anomaly_mask,
            'image': augmented_image,
            'has_anomaly': has_anomaly,
            'idx': idx,
            'anomaly_source_img': anomaly_source_img,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
            'pt_box': boxes,
            'frequency_img': frequency_img,
            'text_prompt': text_prompt
        }

        return sample


class MVTecTestDataset(Dataset):

    def __init__(self, args, root_dir, subclass, resize_shape=None, prompt='click'):
        self.args = args
        self.root_dir = root_dir
        self.subclass = subclass
        self.images = sorted(glob.glob(os.path.join(root_dir, '*/*.png')))
        self.resize_shape = resize_shape
        self.prompt = prompt

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape is not None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # imagegray = cv2.equalizeHist(imagegray.astype(np.uint8))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        # fft
        f = np.fft.fft2(imagegray)
        fshift = np.fft.fftshift(f)

        # BHPF
        rows, cols = imagegray.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        d = 30  # cutoff frequency
        n = 4  # BHPF order
        epsilon = 1e-6  # avoid dividing zero
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
        maska = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
        fshift_filtered = fshift * maska

        # inverse fft
        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        imagegray = np.real(image_filtered).astype(np.float32)
        imagegray = imagegray[:, :, None]

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        imagegray = np.transpose(imagegray, (2, 0, 1))

        return image, mask, imagegray

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask, frequency_img = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask, frequency_img = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        point_label = 1  # 默认的点标签
        num_points = 10

        if self.prompt == 'click' and has_anomaly == 1:
            # 仅在有异常的情况下生成点提示
            point_label, pt = random_click(mask, point_label, num_points)
            pt = pt[:, 1:]
        else:
            point_label = 0
            point_label, pt = random_click_black(mask, point_label, num_points)

        box_num, boxes = find_disconnected_bounding_boxes(mask)

        frequency_img = frequency_img / 128.0

        img_vis = image

        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test_seg = transforms.Compose([
            transforms.Resize((args.out_size, args.out_size))
        ])

        image = torch.tensor(image)
        mask = torch.tensor(mask)

        state = torch.get_rng_state()
        image = transform_test(image)
        torch.set_rng_state(state)

        mask = transform_test_seg(mask).int()

        directory, file_name = os.path.split(img_path)
        name = os.path.basename(directory)
        image_meta_dict = {'filename_or_obj': name + file_name}

        # print(f"Frequency Image - Max: {frequency_img.max()}, Min: {frequency_img.min()}")
        # print(f"Image - Max: {image.max()}, Min: {image.min()}")
        text_prompt = 'defect' if has_anomaly[0] == 1 else 'good'

        sample = {
            'image': image,
            'img_vis': img_vis,
            'has_anomaly': has_anomaly,
            'label': mask,
            'idx': idx,
            'p_label': point_label,
            'pt': pt,  # 新增的 pt 字段用于存储点提示信息
            'image_meta_dict': image_meta_dict,
            'pt_box': boxes,
            'frequency_img': frequency_img,
            'text_prompt': text_prompt
        }

        return sample









# Visualization part remains the same

if __name__ == '__main__':
    args = cfg.parse_args()
    data_path = args.data_path
    obj_name = args.subclass
    source_path = args.anomaly_source_path

    dataset_train = MVTecTrainDataset(args, data_path + obj_name + "/train/good/", source_path, obj_name, resize_shape=[args.image_size, args.image_size])

    for i_batch, sample_batched in enumerate(dataset_train):
        image = sample_batched["ori_image"].numpy()
        augmented_image = sample_batched["image"].numpy()
        anomaly_mask = sample_batched["label"].numpy()
        has_anomaly = sample_batched["has_anomaly"]
        anomaly_source_img = sample_batched["anomaly_source_img"]
        pt = sample_batched["pt"]
        image_name = sample_batched["image_meta_dict"]["filename_or_obj"]
        p_label = sample_batched["p_label"]
        boxes = sample_batched["pt_box"]  # 获取 boxes 信息
        f_img = sample_batched['frequency_img']

        # print(pt)
        # print(boxes)

        # 转换图像和掩码格式
        image = np.transpose(image, (1, 2, 0))
        augmented_image = np.transpose(augmented_image, (1, 2, 0))
        anomaly_mask = np.transpose(anomaly_mask, (1, 2, 0))
        f_img = np.transpose(f_img, (1, 2, 0))

        # 转换图像为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 5, figsize=(15, 5))

        # 显示原始图像
        axs[0].imshow(image)
        axs[0].set_title('Image')
        axs[0].axis('off')

        point_color = 'ro' if p_label.all() == 1 else 'go'

        # 绘制所有 point prompt (用小圆点表示)
        # for point in pt:
        #     axs[2].plot(point[1], point[0], point_color, markersize=4)

        axs[1].imshow(augmented_image)
        axs[1].set_title('Augmented Image')
        axs[1].axis('off')

        # 显示异常掩码
        axs[2].imshow(anomaly_mask, cmap='gray')
        axs[2].set_title('Anomaly Mask')
        axs[2].axis('off')

        # 显示标注的boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
            axs[2].add_patch(rect)

        # 显示异常来源图像
        axs[3].imshow(anomaly_source_img)
        axs[3].set_title('Anomaly Source Img')
        axs[3].axis('off')

        axs[4].imshow(f_img, cmap='gray')
        axs[4].set_title('Frequency Img')
        axs[4].axis('off')

        # 显示图像
        plt.show()


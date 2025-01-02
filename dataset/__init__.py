import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *
from .mvtec import MVTecTrainDataset, MVTecTestDataset
from torch.utils.data._utils.collate import default_collate


def custom_collate_fn(batch):
    for d in batch:
        if not isinstance(d['p_label'], np.ndarray):
            d['p_label'] = np.array([d['p_label']])

    max_label_length = max(len(d['p_label']) for d in batch)

    for d in batch:
        p_label = d['p_label']
        if len(p_label) < max_label_length:
            padding = np.zeros((max_label_length - len(p_label)), dtype=p_label.dtype)
            d['p_label'] = np.concatenate((p_label, padding))

        pt = d['pt']
        if pt.shape[1] < 2:
            padding = np.zeros((pt.shape[0], 2 - pt.shape[1]), dtype=pt.dtype)
            pt = np.hstack((pt, padding))
        elif pt.shape[1] > 2:
            pt = pt[:, :2]
        d['pt'] = pt

    max_box_count = max(len(d['pt_box']) for d in batch)

    for d in batch:
        boxes = np.array(d['pt_box'])

        # Debugging step: print before and after processing
        # print(f"Before processing: {boxes}")

        if len(boxes) < max_box_count:
            padding = np.zeros((max_box_count - len(boxes), 4), dtype=boxes.dtype)
            d['pt_box'] = np.concatenate((boxes, padding), axis=0)
        elif len(boxes) > max_box_count:
            d['pt_box'] = boxes[:max_box_count]

        # print(f"After processing: {d['pt_box']}")  # Debugging output

        # Ensure consistency in types (convert lists to arrays if necessary)
        d['pt_box'] = np.array(d['pt_box'])

    return default_collate(batch)



def get_dataloader_0(args):
    # Define the training subclass
    train_subclass = args.subclass

    # Define the testing subclasses
    test_subclasses = [args.subclass]

    # Load the training data
    mvtec_train_dataset = MVTecTrainDataset(args, args.data_path + train_subclass + "/train/good/",
                                            args.anomaly_source_path, train_subclass,
                                            resize_shape=[args.image_size, args.image_size])
    mvtec_train_loader = DataLoader(mvtec_train_dataset, batch_size=args.b, shuffle=True,
                                    num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    # Load the test data for each subclass and store them in a dictionary
    test_loaders = {}
    for subclass in test_subclasses:
        mvtec_test_dataset = MVTecTestDataset(args, args.data_path + subclass + "/test", subclass,
                                              resize_shape=[args.image_size, args.image_size])
        test_loaders[subclass] = DataLoader(mvtec_test_dataset, batch_size=1, shuffle=False,
                                            num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    return mvtec_train_loader, test_loaders



def get_dataloader(args):
    # Define the training subclass
    train_subclasses = [args.subclass]

    # Define the testing subclasses
    #test_subclasses = [ 'carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
    test_subclasses = [args.subclass]

    train_loaders = {}
    for subclass in train_subclasses:
        mvtec_train_dataset = MVTecTestDataset(args, args.data_path + subclass + "/test", subclass,
                                               resize_shape=[args.image_size, args.image_size])
        train_loaders[subclass] = DataLoader(mvtec_train_dataset, batch_size=args.b, shuffle=True,
                                             num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    # Load the test data for each subclass and store them in a dictionary
    test_loaders = {}
    for subclass in test_subclasses:
        mvtec_test_dataset = MVTecTestDataset(args, args.data_path + subclass + "/test", subclass,
                                              resize_shape=[args.image_size, args.image_size])
        test_loaders[subclass] = DataLoader(mvtec_test_dataset, batch_size=1, shuffle=False,
                                            num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    return train_loaders, test_loaders
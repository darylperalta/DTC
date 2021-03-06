from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.backends.cudnn as cudnn
import random
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate, DataLoader
from utils import TransformTwice, RandomTranslateWithReflect

def find_classes_from_folder(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def find_classes_from_file(file_path):
    with open(file_path) as f:
            classes = f.readlines()
    classes = [x.strip() for x in classes] 
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, classes, class_to_idx):
    samples = []
    for target in classes:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                if 'JPEG' in path or 'jpg' in path:
                    samples.append(item)
    
    return samples 

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def pil_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, transform=None, target_transform=None, samples=None, loader=pil_loader):
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 images in subfolders \n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.samples=samples 
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.samples)

def ImageNetLoader882(batch_size, num_workers, split='train', shuffle=False, path='data_shallow14/datasets/ImageNet/'):
    img_split = 'images/'+split
    classes_118, class_to_idx_118 = find_classes_from_file(path+'imagenet_rand118/imagenet_118.txt')
    samples_118 = make_dataset(path+img_split, classes_118, class_to_idx_118)
    classes_1000, _ = find_classes_from_folder(path+img_split)
    classes_882 = list(set(classes_1000) - set(classes_118))
    class_to_idx_882 = {classes_882[i]: i for i in range(len(classes_882))}
    samples_882 = make_dataset(path+img_split, classes_882, class_to_idx_882)
    if split=='train':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    dataset = ImageFolder(transform=transform, samples=samples_882)
    dataloader_882 = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True) 
    return dataloader_882
    
def ImageNetLoader30(batch_size, num_workers=2, path='data_shallow14/datasets/ImageNet/', subset='A', aug=None, shuffle=False, subfolder='train'):
    # dataloader of 30 classes
    classes_30, class_to_idx_30 = find_classes_from_file(path+'imagenet_rand118/imagenet_30_{}.txt'.format(subset))
    samples_30 = make_dataset(path+'images/{}'.format(subfolder), classes_30, class_to_idx_30)
    if aug == None:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='once':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ]))
    dataset = ImageFolder(transform=transform, samples=samples_30)
    dataloader_30 = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True) 
    return dataloader_30

def ImageNetLoader800from882(batch_size, num_workers, split='train', path='data_shallow14/datasets/ImageNet/'):
    # this dataloader split the 882 classes into train + val = 882
    classes_118, class_to_idx_118 = find_classes_from_file(path+'imagenet_rand118/imagenet_118.txt')
    samples_118 = make_dataset(path+'images/train', classes_118, class_to_idx_118)
    classes_1000, _ = find_classes_from_folder(path+'images/train')
    classes_882 = list(set(classes_1000) - set(classes_118))
    classes_train = classes_882[:800]
    class_to_idx_train = {classes_train[i]: i for i in range(len(classes_train))}
    samples_800 = make_dataset(path+'images/'+split, classes_train, class_to_idx_train)
    if split=='train':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    dataset_800= ImageFolder(transform=transform, samples=samples_800)
    dataloader_800= DataLoader(dataset_800, batch_size=batch_size, shuffle=split=='train', num_workers=num_workers, pin_memory=True) 
    return dataloader_800

def ImageNetLoader82from882(batch_size, num_workers, num_val_cls=30, path='data_shallow14/datasets/ImageNet/'):
    classes_118, class_to_idx_118 = find_classes_from_file(path+'imagenet_rand118/imagenet_118.txt')
    samples_118 = make_dataset(path+'images/train', classes_118, class_to_idx_118)
    classes_1000, _ = find_classes_from_folder(path+'images/train')
    classes_882 = list(set(classes_1000) - set(classes_118))
    classes_val = classes_882[800:800+num_val_cls]
    class_to_idx_val = {classes_val[i]: i for i in range(len(classes_val))}
    samples_val = make_dataset(path+'images/train', classes_val, class_to_idx_val)
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset_val = ImageFolder(transform=transform, samples=samples_val)
    dataloader_val= DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) 
    return dataloader_val

if __name__ == '__main__':
    path = 'data_shallow14/datasets/ImageNet/'
    classes_118, class_to_idx_118 = find_classes_from_file(path+'imagenet_rand118/imagenet_118.txt')
    samples_118= make_dataset(path+'images/train', classes_118, class_to_idx_118)
    classes_1000, _ = find_classes_from_folder(path+'images/train')
    classes_882 = list(set(classes_1000) - set(classes_118))
    class_to_idx_882 = {classes_882[i]: i for i in range(len(classes_882))}
    samples_882 = make_dataset(path+'images/train', classes_882, class_to_idx_882)
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = ImageFolder(transform=transform, samples=samples_882)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers = 2) 
    print('data',len(dataloader))

    dataloader = ImageNetLoader882(batch_size=400, num_workers=2, path='data_shallow14/datasets/ImageNet/')
    print('data882', len(dataloader))
    img, target = next(iter(dataloader))
    print(target)
    dataloader = ImageNetLoader30(batch_size=400, num_workers=2, path='data_shallow14/datasets/ImageNet/', subset='A')
    print('data30', len(dataloader))
    img, target = next(iter(dataloader))
    print(target)


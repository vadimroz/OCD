import torch
import torch.nn as nn
import numpy as np
import os
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from torchvision.datasets import mnist, cifar
from torchvision import transforms
import Lenet5
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from copy import deepcopy
from torchvision import models
from SqueezeNet.squeezenet_model import SqueezeNet
from SqueezeNet.prepare_data import gen_data

def wrapper_dataset(config, args, device):
    if args.datatype == 'tinynerf':

        data = np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] = get_minibatches
        batch['chunksize'] = chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [], []
        for img, tfrom in zip(images, tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = mnist.MNIST("\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST("\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [], []
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:, 0, :, :].unsqueeze(1)
            batch = {'input': train_x, 'output': train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:, 0, :, :].unsqueeze(1)
            batch = {'input': train_x, 'output': train_label}
            test_ds.append(deepcopy(batch))
    elif args.datatype == 'cifar10':
        model = models.__dict__['densenet'](
            num_classes=10,
            depth=100,
            growthRate=12,
            compressionRate=2,
            dropRate=0,
        )
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = cifar.CIFAR10(root=os.path.join(os.getcwd(),"OCD/data/"), train=True, download=True, transform=transform_train)
        train_loader = DataLoader(trainset, batch_size=1, shuffle=True)
        testset = cifar.CIFAR10(root=os.path.join(os.getcwd(),"OCD/data/"), train=False, download=True, transform=transform_test)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)
        train_ds, test_ds = [], []
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch = {'input': inputs, 'output': targets}
            test_ds.append(deepcopy(batch))
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch = {'input': inputs, 'output': targets}
            train_ds.append(deepcopy(batch))
    elif args.datatype == 'CTSD':
        train_ds, test_ds = [], []
        model = SqueezeNet(num_classes=58)
        dataloaders = gen_data()
        for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
            batch = {'input': inputs, 'output': targets}
            train_ds.append(deepcopy(batch))
        for batch_idx, (inputs, targets) in enumerate(dataloaders['test']):
            batch = {'input': inputs, 'output': targets}
            test_ds.append(deepcopy(batch))
    return train_ds, test_ds, model
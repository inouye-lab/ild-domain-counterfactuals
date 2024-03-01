import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from itertools import permutations, combinations_with_replacement
import random
from sc2image.dataset import StarCraftMNIST

COLOR_MAP = [[1, 2], [0, 1], [0, 2], [2], [1], [0],[]]

class MnistColorRotated(data_utils.Dataset):
    def __init__(self,
                 list_domains=['0', '90'],
                 root='../data',
                 train=True,
                 transform=None,
                 download=True,
                 return_color=False):

        self.list_domains = list_domains
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.download = download
        self.return_color = return_color

        assert self.list_domains == ['0', '90'], 'Do not support other domain setup yet!'

        self.imgs, self.label, self.domain, self.colors = self._get_data()


    def _get_data(self):
        # =================================================================================== #
        #                         Load MNIST and get subset                                   #
        # =================================================================================== #
        dataset = datasets.MNIST(self.root, train=self.train,
                                 download=self.download,
                                 transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        for i, (x, y) in enumerate(loader):
            img_og = x
            label_og = y
        img_og = img_og[(label_og==0)|(label_og==1)|(label_og==2)]
        label_og = label_og[(label_og==0)|(label_og==1)|(label_og==2)]
        #label_og -= 1
        #label_og = torch.where(label_og == 2,1,0, torch.where(label_og == 3,2,1, label_og))

        # =================================================================================== #
        #                         Get rotated images                                          #
        # =================================================================================== #
        img_rot = []
        label_rot = []
        domain_rot = []
        for d, rotation in enumerate(self.list_domains):
            if rotation == '0':
                img_rot.append(img_og)
            else:
                img_rot.append(transforms.functional.rotate(img_og, int(rotation)))

            label_rot.append(label_og)
            domain_rot.append(torch.ones(label_og.size()) * d)

        img_rot = torch.cat(img_rot)
        label_rot = torch.cat(label_rot)
        domain_rot = torch.cat(domain_rot)

        # =================================================================================== #
        #                         Get colored images                                          #
        # =================================================================================== #

        img_col = []
        label_col = []
        domain_col = []
        color_col = []
        img_rot = img_rot.repeat(1, 3, 1, 1)

        for dd in range(2):
            for yy in range(3):
                chosen_indices = ((domain_rot == dd) & (label_rot == yy))
                img_col_temp = img_rot[chosen_indices]
                #indices_main_color = torch.bernoulli(torch.ones(len(img_col_temp)) * 0.9).bool()
                color_idx = dd*2+yy
                color_labels = torch.multinomial(torch.Tensor([0.5 if i==color_idx else 0.5/5 for i in range(6)]),num_samples=len(img_col_temp), replacement=True)
                for cdx,color in enumerate(color_labels):
                    img_col_temp[cdx,COLOR_MAP[color]] = 0
                img_col.append(img_col_temp)

                # add label and domain
                label_col.append(label_rot[chosen_indices])
                domain_col.append(domain_rot[chosen_indices])

                # add color
                color_col.append(color_labels)

        img_col = torch.cat(img_col)
        label_col = torch.cat(label_col)
        domain_col = torch.cat(domain_col)
        color_col = torch.cat(color_col)

        return img_col, label_col.long(), domain_col.long(), color_col.long()

    def __len__(self):

        return len(self.label)

    def __getitem__(self, index):

        x = self.imgs[index]
        y = self.label[index]
        d = self.domain[index]
        if self.return_color:
            c = self.colors[index]

        if self.transform is not None:
            x = self.transform(x)
        if self.return_color:
            return x, y, d, c
        else:
            return x, y, d

class MnistRotated(data_utils.Dataset):
    def __init__(self,
                 list_domains=['0', '15', '30', '45', '60'],
                 root='../data',
                 train=True,
                 transform=None,
                 download=True,
                 mnist_type='mnist',
                 subsample=False):

        self.list_domains = list_domains
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.download = download
        self.mnist_type = mnist_type
        self.subsample = subsample

        self.imgs, self.label, self.domain = self._get_data()


    def _get_data(self):
        # =================================================================================== #
        #                         Load MNIST and get subset                                   #
        # =================================================================================== #
        if self.mnist_type == 'rmnist':
            dataset = datasets.MNIST(self.root, train=self.train,
                                     download=self.download,
                                     transform=transforms.ToTensor())
        elif self.mnist_type == 'rfmmnist':
            dataset = datasets.FashionMNIST(self.root, train=self.train,
                                     download=self.download,
                                     transform=transforms.ToTensor())
        elif self.mnist_type == 'rscmnist':
            dataset = StarCraftMNIST(self.root, train=self.train,
                                     download=self.download,
                                     transform=transforms.ToTensor())
        else:
            raise ValueError('Unknown mnist type: {}'.format(self.mnist_type))

        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        for i, (x, y) in enumerate(loader):
            img_og = x
            label_og = y

        print(img_og.shape)

        # =================================================================================== #
        #                         Subsample if needed                                         #
        # =================================================================================== #
        if self.subsample:
            if self.train:
                class_size = 600
            else:
                class_size = 100
            subsample_indices = []
            for i in range(10):
                class_indices = torch.where(label_og == i)[0]
                subsample_indices.append(class_indices[torch.randperm(class_indices.shape[0])][:class_size])
            subsample_indices = torch.cat(subsample_indices)
            img_og = img_og[subsample_indices]
            label_og = label_og[subsample_indices]

        print(img_og.shape)

        # =================================================================================== #
        #                         Get rotated images                                          #
        # =================================================================================== #
        img_rot = []
        label_rot = []
        domain_rot = []
        for d, rotation in enumerate(self.list_domains):
            if rotation == '0':
                img_rot.append(img_og)
            else:
                img_rot.append(transforms.functional.rotate(img_og, int(rotation)))

            label_rot.append(label_og)
            domain_rot.append(torch.ones(label_og.size()) * d)

        img_rot = torch.cat(img_rot)
        label_rot = torch.cat(label_rot)
        domain_rot = torch.cat(domain_rot)

        return img_rot, label_rot.long(), domain_rot.long()

    def __len__(self):

        return len(self.label)

    def __getitem__(self, index):

        x = self.imgs[index]
        y = self.label[index]
        d = self.domain[index]

        if self.transform is not None:
            x = self.transform(x)
        return x, y, d
r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from data.pascal import DatasetPASCAL
from data.fss import DatasetFSS
from data.deepglobe import DatasetDeepglobe
from data.isic import DatasetISIC
from data.lung import DatasetLung
#FDG2801
from data.isic2017 import DatasetISIC2017

# Third DA
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'deepglobe': DatasetDeepglobe,
            'isic': DatasetISIC,
            'lung': DatasetLung,
            'isic2017': DatasetISIC2017,
        }
        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        #### Original
        # cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(cls.img_mean, cls.img_std)])
        ##### New data augmentation for tests
        # cls.transform = transforms.Compose([
        #                     transforms.Resize(size=(512, 512)),  # Nota: in torchvision, Resize accetta una tupla (height, width)
        #                     transforms.ToTensor(),
        #                     transforms.Normalize(
        #                         mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225]
        #                     )
        #             ])
        #### Second data augmentation test
        # cls.transform = transforms.Compose([
        #                     transforms.Resize(size=(512, 512)),  # Nota: in torchvision, Resize accetta una tupla (height, width)
        #                     transforms.ToTensor(),
        #                     transforms.Normalize(
        #                         mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225],
        #                         max_pixel_value = 255.0,
        #                         p = 1.0
        #                     )
        #             ])
        #### Third data augmentation test
        cls.transform = A.Compose([
                        A.Resize(width=224,height=224),
                        ToTensorV2(),
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value=255.0,
                            p=1.0
                        )], p=1.)

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == 'trn' else 0
        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=bsz, num_workers=nworker, drop_last=True, sampler=sampler, pin_memory=True)
        return dataloader, sampler

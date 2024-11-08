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
import random
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
        # if data_aug==1:
        #     cls.img_mean = [0.485, 0.456, 0.406]
        #     cls.img_std = [0.229, 0.224, 0.225]
        # else:
        #     if data_aug==2:
        #         cls.img_mean=[0.684, 0.483, 0.519],
        #         cls.img_std=[0.229, 0.224, 0.225]
        #     else: 
        #         if data_aug==3:
        #             cls.img_mean=[0.763, 0.545, 0.570],
        #             cls.img_std=[0.140, 0.152, 0.169]
        #         else:
        #             if data_aug==4:
        #                 cls.img_mean=[0.684, 0.483, 0.519],
        #                 cls.img_std=[0.185, 0.186, 0.199]
        cls.datapath = datapath
        #### Original
        # cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(cls.img_mean, cls.img_std)])
        # #ISIC 2017 con ISIC 2017 Model; 25-27 mIoU; 26-27 FBIoU; lr 1e-4
        # ---------------------------------------------------------------------------------------#
        '''
        NB: dove non specificato, mean ed std hanno i valori di (rispettivamente) [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        +++ ISIC 2018 DATASET +++
        1 SHOT
        +++ COV MODEL
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-4; 23.18 mIoU; 12.88 FBIoU; imgsize 300x300
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-6; 22.91 mIoU; 11.45 FBIoU; imgsize 300x300
            ---
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-4; 21.94 mIoU, 47.78 FBIoU; imgsize 300x300
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-6; 18.29 mIoU, 36.27 FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.229, 0.224, 0.225]
            ---
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-4; 18.55 mIoU, 20.53 FBIoU; img size 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.229, 0.224, 0.225]
            ---
        5 SHOT
            ISIC 2018; COV MODEL; 5 SHOT; LR 1E-4; 21.86 mIoU, 12.78 FBIoU; imgsize 300x300
            ISIC 2018; COV MODEL; 5 SHOT; LR 1E-6; 26.51 mIoU, 42.83 FBIoU; imgsize 300x300
            ---
            ISIC 2018; COV MODEL; 5 SHOT; LR 1E-6; 27.23 mIoU, 37.70 FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.229, 0.224, 0.225]
            ---
            ISIC 2018; COV MODEL; 5 SHOT; LR 1E-4; 00.00 mIoU, 35.0+ FBIoU; imgsize 300x300
            ISIC 2018; COV MODEL; 5 SHOT; LR 1E-3; 24.10 mIoU, 20.22 FBIoU; imgsize 300x300
            ISIC 2018; COV MODEL; 5 SHOT; LR 1E-5; 25.65 mIoU, 25.44 FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.229, 0.224, 0.225]
            ---
        
        +++ ISIC 2018 MODEL
        1 SHOT
            ----
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-3; 23.93 mIoU; 15.94 FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-4; 25.41 mIoU; 16.71 FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-6; 23.13 mIoU; 12.29 FBIoU; imgsize 300x300
            ---
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-4; 25.18 mIoU; 20.64 FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-6; 22.91 mIoU, 11.46 FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-6; 24.57 mIoU, 18.06 FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.229, 0.224, 0.225]
            ---
        5 SHOT
        ----
            ISIC 2018; ISIC 2018 MODEL; 5 SHOT; LR 1E-4; 21.41 mIoU; 10.70 FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 5 SHOT; LR 1E-6; 21.74 mIoU; 26.65 FBIoU; imgsize 300x300
            ---
            ISIC 2018; ISIC 2018 MODEL; 5 SHOT; LR 1E-4; 24.06 mIoU; 22.74 FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 5 SHOT; LR 1E-6; 26.46 mIoU, 15.74 FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2018; ISIC 2018 MODEL; 5 SHOT; LR 1E-6; 25.79 mIoU, 12.52 FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.229, 0.224, 0.225]
            ---
---------------------------------------------------------------------------------------------------------------------------------------------------------
        +++ ISIC 2017 DATASET +++
        +++ COV MODEL
        1 SHOT
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; 25.56 mIoU; 12.97 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; 25.55 mIoU; 12.88 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; 25.51 mIoU; 12.76 FBIoU; imgsize 300x300
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; 25.74 mIoU; 13.54 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; 17.97 mIoU; 23.74 FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; 24.45 mIoU; 15.99 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; 28.65 mIoU; 36.56 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; 25.51 mIoU, 12.76 FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.229, 0.224, 0.225]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; 25.89 mIoU; 15.11 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; 26.00 mIoU, 14.41 FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.185, 0.186, 0.199]
            ---
        5 SHOT
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-3; 25.75 mIoU; 13.02 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-4; 26.31 mIoU; 45.10 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-6; 27.76 mIoU; 19.63 FBIoU; imgsize 300x300
            ---
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-4; 30.05 mIoU; 14.42 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-6; 19.42 mIoU; 41.64 FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-3; 25.75 mIoU; 16.89 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-4; 25.61 mIoU; 12.88 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-6; 25.68 mIoU, 16.88FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.229, 0.224, 0.225]
            ---
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-3; 26.73 mIoU; 16.22 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-6; 26.66 mIoU, 16.00 FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.185, 0.186, 0.199]
            ---
        +++ ISIC 2017 MODEL
        1 SHOT
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; 28.00 mIoU; 38.83 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; 25.51 mIoU; 12.76 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-5; 25.16 mIoU; 30.13 FBIoU; imgsize 300x300
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-5; 25.51 mIoU; 12.76 FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; 25.92 mIoU; 15.44 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; 26.04 mIoU, 14.69 FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.229, 0.224, 0.225]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; 22.68 mIoU; 43.31 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; 25.51 mIoU; 12.76 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; 24.06 mIoU, 30.91 FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.185, 0.186, 0.199]
            ---
        5 SHOT
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-3; 32.20 mIoU; 14.66 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-4; 25.75 mIoU; 12.88 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-5; 26.36 mIoU; 15.07 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-6; 26.10 mIoU; 14.80 FBIoU; imgsize 300x300
            ---
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-5; 26.00 mIoU, 13.20FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-6; 30.05 mIoU, 15.60 FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.229, 0.224, 0.225]
            ---
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-4; 29.00 mIoU; 27.00 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 5 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            mean=[0.684, 0.483, 0.519],
            std=[0.185, 0.186, 0.199]
        '''
        def clamp_to_max_pixel_value(tensor, max_pixel_value=255.0):
            return torch.clamp(tensor * max_pixel_value, 0, max_pixel_value) / max_pixel_value

        cls.transform = transforms.Compose([
            transforms.Resize(size=(300, 300)),  
            #transforms.ToTensor(),
            transforms.RandomCrop(size=(256, 256)),  
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),   
            transforms.RandomRotation(degrees=(-15, 15)), 
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.5))], p=0.4),  
            transforms.RandomApply([
                transforms.Lambda(lambda img: transforms.functional.adjust_gamma(img, gamma=random.uniform(0.7, 1.3))),  
                transforms.Lambda(lambda img: transforms.functional.adjust_sharpness(img, sharpness_factor=random.uniform(0.8, 1.2))) 
            ], p=0.3), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Lambda(lambda x: clamp_to_max_pixel_value(x, max_pixel_value=255.0))
        ])


    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == 'trn' else 0
        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=bsz, num_workers=nworker, drop_last=True, sampler=sampler, pin_memory=True)
        return dataloader, sampler

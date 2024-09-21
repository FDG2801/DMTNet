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
        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
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
            # mean=[0.684, 0.483, 0.519],
            # std=[0.229, 0.224, 0.225]
            ---
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-4; 18.55 mIoU, 20.53 FBIoU; img size 300x300
            # mean=[0.763, 0.545, 0.570],
            # std=[0.229, 0.224, 0.225]
            ---
        5 SHOT
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-4; mIoU, FBIoU; imgsize 300x300
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-6; mIoU; FBIoU; imgsize 300x300
            ---
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-4; mIoU, FBIoU; imgsize 300x300
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.229, 0.224, 0.225]
            ---
            ISIC 2018; COV MODEL; 1 SHOT; LR 1E-4; mIoU, FBIoU; imgsize 300x300
            # mean=[0.763, 0.545, 0.570],
            # std=[0.229, 0.224, 0.225]
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
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-6; mIoU; FBIoU; imgsize 300x300
            ---
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2018; ISIC 2018 MODEL; 1 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.229, 0.224, 0.225]
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
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-5; 17.97 mIoU; 23.74 FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; 24.45 mIoU; 15.99 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; 28.65 mIoU; 36.56 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; 25.51 mIoU, 12.76 FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.229, 0.224, 0.225]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; 25.89 mIoU; 15.11 FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; 26.00 mIoU, 14.41 FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.185, 0.186, 0.199]
            ---
        5 SHOT
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; mIoU; FBIoU; imgsize 300x300
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.229, 0.224, 0.225]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.185, 0.186, 0.199]
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
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; mIoU; FBIoU; imgsize 300x300
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            mean=[0.763, 0.545, 0.570],
            std=[0.140, 0.152, 0.169]
            ---
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-3; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-4; mIoU; FBIoU; imgsize 300x300
            ISIC 2017; ISIC 2017 MODEL; 1 SHOT; LR 1E-6; mIoU, FBIoU; imgsize 300x300
            # mean=[0.684, 0.483, 0.519],
            # std=[0.229, 0.224, 0.225]
        '''
        def clamp_to_max_pixel_value(tensor, max_pixel_value=255.0):
            return torch.clamp(tensor * max_pixel_value, 0, max_pixel_value) / max_pixel_value

        cls.transform = transforms.Compose([
            transforms.Resize(size=(300, 300)),  # Aumenta leggermente la dimensione iniziale per preservare più dettagli
            #transforms.ToTensor(),
            transforms.RandomCrop(size=(256, 256)),  # Ritaglia casualmente per variare la posizione della lesione e aumentare il dataset
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),   # Aumenta leggermente la probabilità di flip verticale
            transforms.RandomRotation(degrees=(-15, 15)), # Aumenta leggermente l'intervallo di rotazione
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # Aumenta leggermente la gamma di variazioni di colore
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.5))], p=0.4),  # Aumenta leggermente la probabilità e la gamma di sfocatura
            transforms.RandomApply([
                transforms.Lambda(lambda img: transforms.functional.adjust_gamma(img, gamma=random.uniform(0.7, 1.3))),  # Aumenta la gamma di variazioni di gamma
                transforms.Lambda(lambda img: transforms.functional.adjust_sharpness(img, sharpness_factor=random.uniform(0.8, 1.2))) # Aggiungi variazioni di nitidezza
            ], p=0.3), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # ISIC2017: [0.763, 0.545, 0.570], [0.684, 0.483, 0.519]; 
                std=[0.229, 0.224, 0.225] # ISIC2017:  [0.140, 0.152, 0.169], [0.185, 0.186, 0.199]; 
                # mean=[0.684, 0.483, 0.519],
                # std=[0.229, 0.224, 0.225]
            ),
            transforms.Lambda(lambda x: clamp_to_max_pixel_value(x, max_pixel_value=255.0))
        ])



        # cls.transform = transforms.Compose([
        #     transforms.Resize(size=(320, 320)),  
        #     transforms.RandomCrop(size=(256, 256)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5), 
        #     transforms.RandomRotation(degrees=(-20, 20)), 
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        #     transforms.RandomApply([
        #         transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0)),
        #         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, fill=0),
        #         transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        #         transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0), 
        #     ], p=0.5), 
        #     transforms.RandomApply([
        #         transforms.Lambda(lambda img: transforms.functional.adjust_gamma(img, gamma=random.uniform(0.6, 1.4))),
        #         transforms.Lambda(lambda img: transforms.functional.adjust_sharpness(img, sharpness_factor=random.uniform(0.7, 1.3))),
        #         transforms.RandomPosterize(bits=4, p=0.3),
        #         transforms.RandomEqualize(p=0.2),
        #         transforms.RandomSolarize(threshold=128, p=0.2)
        #     ], p=0.4), 
        #     transforms.ToTensor(), # Applicato prima della normalizzazione
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     ),
        #     transforms.Lambda(lambda x: clamp_to_max_pixel_value(x, max_pixel_value=255.0)) 
        # ])


    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == 'trn' else 0
        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=bsz, num_workers=nworker, drop_last=True, sampler=sampler, pin_memory=True)
        return dataloader, sampler

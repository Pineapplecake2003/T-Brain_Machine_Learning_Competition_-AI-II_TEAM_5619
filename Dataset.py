import os
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import processbar

class NavigationDataset(Dataset):
    def __init__(self, mode, dataset_path:str, test_mode=None) -> None:
        self.mode = mode.lower()
        self.test_mode = test_mode
        self.datas = []
        self.check_transform = transforms.ToPILImage()
        
        self.data_transform = transforms.Compose([
            transforms.Resize((192, 192), antialias=True, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.9, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        self.ground_truth_transform = transforms.Compose([
            transforms.Resize((192, 192), antialias=True, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        
        self.test_data_transform = transforms.Compose([
            transforms.Resize((192, 192), antialias=True, interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        if mode.lower() == "train":
            data_path = dataset_path + "./Training_dataset/img/"
            ground_truth_path = dataset_path + "./Training_dataset/label_img/"
            for nth in range(len(os.listdir(data_path))):
                if nth < 2160:
                    img_file = f"TRA_RI_{2000000 + nth}"
                else:
                    img_file = f"TRA_RO_{2000000 + nth}"
                
                self.datas.append((
                    Image.open(data_path + img_file + ".jpg").convert("RGB"),
                    Image.open(ground_truth_path + img_file + ".png").convert("L")
                ))
                processbar(nth + 1, len(os.listdir(data_path)), total_len = 30, info = f"{nth+1:5d}/{len(os.listdir(data_path)):5d} image loaded.")
        elif mode.lower() == "test":
            data_path = dataset_path + "./Testing_dataset/"
            for nth in range(len(os.listdir(data_path))):
                if nth < 360:
                    img_file = f"{self.test_mode}_RI_{2000000 + nth}"
                else:
                    img_file = f"{self.test_mode}_RO_{2000000 + nth}"
                self.datas.append((Image.open(data_path + img_file + ".jpg").convert("RGB")))
                processbar(nth + 1, len(os.listdir(data_path)), total_len = 30, info = f"{nth+1:5d}/{len(os.listdir(data_path)):5d} image loaded.")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.mode == "train":
            angle = random.sample([0, 90, 180, 270], 1)[0]
            
            seed = random.random() * 100
            torch.manual_seed(seed)
            data = F.rotate(self.data_transform(self.datas[index][0]), angle)
            
            torch.manual_seed(seed)
            ground_truth = F.rotate(self.ground_truth_transform(self.datas[index][1]), angle)
        else:
            data = self.test_data_transform(self.datas[index])
            
            ground_truth = data
        return (data, ground_truth)

class ValidationDataset(Dataset):
    def __init__(self, dataset_path:str) -> None:
        self.datas = []

        self.test_data_transform = transforms.Compose([
            transforms.Resize((192, 192), antialias=True, interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        data_path = dataset_path + "./Training_dataset/img/"
        
        for nth in range(len(os.listdir(data_path))):
            if nth < 2160:
                img_file = f"TRA_RI_{2000000 + nth}"
            else:
                img_file = f"TRA_RO_{2000000 + nth}"
            
            self.datas.append((
                Image.open(data_path + img_file + ".jpg").convert("RGB")
            ))
            processbar(nth + 1, len(os.listdir(data_path)), total_len = 30, info = f"{nth+1:5d}/{len(os.listdir(data_path)):5d} image loaded.")
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.test_data_transform(self.datas[index])
        return (data, data)


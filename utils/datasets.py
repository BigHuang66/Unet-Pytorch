import torch
import os
import cv2
import json
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(256, 256), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ]
)

class CustomDataset(Dataset):
    def __init__(self, data_path, image_set_file, transform=None, ignore_index=255):
        self.image_path = os.path.join(data_path, 'images')
        self.label_path = os.path.join(data_path, 'labels')
        self.transform = transform
        self.num_classes = 3
        self.ignore_index = ignore_index
        self.file_list = []
        if image_set_file.endswith('.json'):
            flist = json.load(open(os.path.join(data_path, image_set_file)))
            self.file_list = list(flist.keys())
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = cv2.imread(os.path.join(self.image_path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.label_path, img_name.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]
        return {"image": image, "label": label}

if __name__ == "__main__":
    data_dir = "./data/caries"
    train_set = 'train.json'
    val_set = 'val.json'

    train_data = CustomDataset(data_dir, image_set_file=train_set, transform=train_transform)
    valid_data = CustomDataset(data_dir, image_set_file=val_set, transform=val_transform)
    print(len(train_data), len(valid_data))

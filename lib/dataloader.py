from PIL import Image
from os import path
from torchvision import transforms
from torch.utils import data
import torch


class CelebADataset(data.Dataset):
    def __init__(self, base_path, data_name_list, label_dict=None, image_size=64, augment_prob=float(0)):
        self._data_name_list = data_name_list
        self._base_path = base_path
        self._label_dict = label_dict
        self._transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            # # Here are some data augmentation tools may be useful
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0),
            # transforms.RandomHorizontalFlip(p=augment_prob),
            # transforms.RandomVerticalFlip(p=augment_prob),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        image_name = self._data_name_list[index]
        image_path = path.join(self._base_path, image_name)
        image = Image.open(image_path)
        image = self._transform(image)
        if self._label_dict is None:
            return image, index, image_name
        else:
            image_label = self._label_dict[image_name]
            image_label = torch.IntTensor(image_label)
            return image, index, image_name, image_label

    def __len__(self):
        return len(self._data_name_list)

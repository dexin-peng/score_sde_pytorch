'''
Mainly based on https://github.com/yang-song/score_sde_pytorch/blob/main/datasets.py
'''

import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from functools import cached_property

class BaseDataLoader:
    def __init__(self, config):
        self.config = config

    def data_scaler(self, x):
        if self.config.data.centered:
            return x * 2 - 1
        else:
            return x

    def data_inverse_scaler(self, x):
        if self.config.data.centered:
            return (x + 1) / 2
        else:
            return x

    def crop_resize(self, image, resolution):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        w, h = image.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        image = image.resize((resolution, resolution), Image.BICUBIC)
        return image

    def resize_small(self, image, resolution):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        w, h = image.size
        ratio = resolution / min(w, h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))
        image = image.resize((new_w, new_h), Image.BICUBIC)
        return image

    def central_crop(self, image, size):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        w, h = image.size
        left = (w - size) // 2
        top = (h - size) // 2
        return image.crop((left, top, left + size, top + size))

    @property
    def transform(self, uniform_dequantization=False):

        self.batch_size = self.config.training.batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.batch_size % num_devices != 0:
            raise ValueError(f'Batch size {self.batch_size} must be divisible by the number of devices {num_devices}')

        transform_list = []
        transform_list.append(transforms.Resize(
            (self.config.data.image_size, self.config.data.image_size),
            interpolation=transforms.InterpolationMode.BICUBIC))
        if self.config.data.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        
        if uniform_dequantization:
            transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        return transforms.Compose(transform_list)
    
class DataLoaderRegistry(dict):
    def __getitem__(self, key):
        if not isinstance(key, str):
            name = key.data.name
            return super().__getitem__(name)(key)
        return super().__getitem__(key)
    
    def __call__(self, config):
        return self[config]

_DATA_LOADERS = DataLoaderRegistry()

def register_data_loader(cls=None, *, name=None):
    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _DATA_LOADERS:
            raise ValueError(f'Already registered data loader with name: {local_name}')
        _DATA_LOADERS[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register


@register_data_loader(name='CIFAR10')
class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def train_dataset(self):
        return datasets.CIFAR10(root=self.config.io.in_dataset_path, train=True, download=True, transform=self.transform)
    
    @cached_property
    def eval_dataset(self):
        return datasets.CIFAR10(root=self.config.io.in_dataset_path, train=False, download=False, transform=self.transform)
    
    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.eval_dataset])
    
    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config.data.num_workers, pin_memory=True)
    
    @cached_property
    def eval_loader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.config.data.num_workers, pin_memory=True)
    
    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.config.data.num_workers, pin_memory=True)
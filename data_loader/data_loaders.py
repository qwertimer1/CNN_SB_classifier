import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader

#from .image_dataset import image_dataset
from .dataset_class import image_dataset, audio_dataset




def get_data_loader(config):
    """Returns data loader as specified in configuration."""
    loader_type = config['data_loader']['type']

    if loader_type == 'MnistDataLoader':
        return MnistDataLoader(config)
    elif loader_type == 'ImageDataLoader':
        return ImageDataLoader(config)
    else:
        raise NotImplementedError(f"Loader {loader_type} not implemented.")


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, config):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = config['data_loader']['data_dir']
        self.dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, config)
 
 
class ImageDataLoader(BaseDataLoader):
    """
    image_dataset data loading using BaseDataLoader
    """
    def __init__(self, config):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = config['data_loader']['data_dir']
        self.img_path = config['data_loader']['img_path']
        self.img_filename = config['data_loader']['img_filename']
        self.label_filename = config['data_loader']['label_filename']
        self.imagedataset = config['data_loader']['image_dataset']
        self.dataset = self.image_dataset(self.data_dir, self.img_path, self.img_filename, self.label_filename, transform=None)
        super(ImageDataLoader, self).__init__(self.dataset, config)
        


class AudioDataloader(BaseDataLoader):
    """
    Standard Audio dataloader from file
    """
    def __init__(self, config):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = config['data_loader']['data_dir']
        self.dataset = audio_dataset()
        super(AudioDataLoader, self).__init__(self.dataset, config)

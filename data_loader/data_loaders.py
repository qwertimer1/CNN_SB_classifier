import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader
from . import image_dataset, 


class ImageDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, config):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = config['data_loader']['data_dir']
        self.dataset = image_dataset
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
        self.dataset = audio_dataset
        super(AudioDataLoader, self).__init__(self.dataset, config)
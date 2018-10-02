from torch.utils.data import Dataset
data_dir = '/home/tim/Documents/Masters/CNN_SB_classifier/datasets/Seabed Images/'

data_dir = "D:\datasets\Seabed Images"
import glob
import re
import hashlib
import torch
from torch.utils.data import Dataset

from PIL import Image
import os


class util():
    def as_bytes(bytes_or_text, encoding='utf-8'):
        import six as _six

        """Converts either bytes or unicode to `bytes`, using utf-8 encoding for text.
        Args:
        bytes_or_text: A `bytes`, `str`, or `unicode` object.
        encoding: A string indicating the charset for encoding unicode.
        Returns:
        A `bytes` object.
        Raises:
        TypeError: If `bytes_or_text` is not a binary or unicode string.
        """
        if isinstance(bytes_or_text, _six.text_type):
            return bytes_or_text.encode(encoding)
        elif isinstance(bytes_or_text, bytes):
            return bytes_or_text
        else:
            raise TypeError('Expected binary or unicode string, got {}'.format(bytes_or_text,))




class image_dataset(Dataset):
    def __init__(self, data_dir, img_path, img_filename, label_filename,exp_phase = "train", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage
        self.exp_phase = exp_phase
        # reading img file from file
        self.results = create_image_lists()
        convert_for_pytorch()
        self.label = labels
        

    def __getitem__(self, index):
        
        img = Image.open(os.path.join(self.data_dir, self.results[''][self.exp_phase])
        
        
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label
                         
    def __len__(self):
        return len(self.img_filename)
                           
    def convert_for_pytorch(self):
        images = []
        for sub_dirs in self.sub_dirs:
            self.images.append(results[sub_dir][self.exp_phase])
                         
            
        
       
    def create_image_lists(self):
        MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 -1

        as_bytes = util.as_bytes()
                             
        if not os.path.exists(self.data_dir):
            print("error")
            return None
        result = {}
        self.sub_dirs = [x[0] for x in os.walk(self.data_dir)]
        print(self.sub_dirs)
        is_root_dir = True
        for sub_dir in self.sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == self.data_dir:
                continue
            for extension in extensions:
                file_glob = os.path.join(self.data_dir, dir_name, '*' + extension)
                file_list.extend(glob.glob(file_glob))
            if not file_list:
                print("No files found")
                continue
            if len(file_list) < 20:
                print("folder has less than 20 images, this may cause issues")
            if len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
                print("folder has too many images")
            self.label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images = []
            testing_images = []
            validation_images = []
            for file_name in file_list:
                base_name = os.path.basename(file_name)
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                hash_name_hashed = hashlib.sha1(as_bytes(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) %
                                    (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                                   (100.0 / MAX_NUM_IMAGES_PER_CLASS))
                if percentage_hash < self.validation_percentage:
                    self.validation_images.append(base_name)
                elif percentage_hash < (self.testing_percentage + self.validation_percentage):
                    self.testing_images.append(base_name)
                else:
                    self.training_images.append(base_name)
                result[self.label_name] = {
                    'dir': dir_name,
                    'training': self.training_images,
                    'testing': self.testing_images,
                    'validation': self.validation_images,
                }
        return result
    
class audio_dataset(Dataset):
    """
    NEED TO UPDATE TO AUDIO DATASET
    """
    
    def __init__(self, data_path, path, filename, label_filename, transform=None):
        self.audio_path = os.path.join(data_path, path)
        self.transform = transform
        # reading img file from file
        filepath = os.path.join(data_path, filename)
        fp = open(filepath, 'r')
        self.filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label
    def __len__(self):
        return len(self.img_filename)

    
class genericDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, path):
        """
        Args:
            data_path (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data, self.labels = torch.load(os.path.join(data_path, path))
       
        self.data, self.labels = map(torch.FloatTensor, (self.data, self.labels))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
        

from torch.utils.data import Dataset

class image_dataset(Dataset):
    def __init__(self, data_path, img_path, img_filename, label_filename, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label
    def __len__(self):
        return len(self.img_filename)
    
    
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
        

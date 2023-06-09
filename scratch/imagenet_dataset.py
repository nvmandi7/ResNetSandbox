
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

# See TORCHVISION.DATASETS.IMAGENET to add more details if desired

class ImageNetDataset(Dataset):

    '''
    Args expected on init:
     - root_dir: Path to directory containing the images
     - annotations_csv: Path to the csv file containing annotations. Necessary columns
        - image_path: Path to the image file
        - label: Label of the image
    
    '''
    
    def __init__(self, root_dir: str, annotations_csv: str) -> None:
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_csv)
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index: int):
        image_path = os.path.join(self.root_dir, self.annotations.image_path.iloc[index])
        image = torch.load(image_path)
        y_label = torch.tensor(int(self.annotations.label.iloc[index]))
        
        return (image, y_label)
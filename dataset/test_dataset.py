import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, RandomRotation

class GraphemeDataSet(Dataset):

    def __init__(self, root_dir, csv_file):
        
        df = pd.read_parquet(root_dir + csv_file)

        self.image_ids = df['image_id'].values
        self.image_array = df.iloc[:, 1:].values

        #transforms
        self.transform = transforms.Compose([
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        self.root_dir = root_dir

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_id = self.image_ids[index]
        image = self.image_array[index, :]
        image = image.reshape(137, 236).astype(np.float32)
        image = Image.fromarray(image).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return {
            "image" : torch.tensor(image, dtype=torch.float),
            "image_id" : image_id
        }        
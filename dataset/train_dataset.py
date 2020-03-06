import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, RandomRotation

class GraphemeDataSet(Dataset):

    def __init__(self, root_dir, csv_file, fold = 0):
        
        df = pd.read_csv(root_dir + csv_file)
        df = df[df['kfold'].isin(fold)].reset_index(drop=True)

        self.image_ids = df['image_id'].values
        self.y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]

        #transforms
        if len(fold) == 1: #fvalidation set
            self.transform = transforms.Compose([
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else: #train set
            self.transform = transforms.Compose([
                RandomRotation((0 ,5)),
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
        image = joblib.load("{}image_pickle/{}.pkl".format(self.root_dir, image_id))
        image = image.reshape(137, 236).astype(np.float32)
        image = Image.fromarray(image).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        y = self.y.iloc[index, :]

        return {
            "image" : torch.tensor(image, dtype=torch.float),
            "grapheme_root" : torch.tensor(y['grapheme_root']).type(torch.LongTensor),
            "vowel_diacritic" : torch.tensor(y['vowel_diacritic']).type(torch.LongTensor),
            "consonant_diacritic" : torch.tensor(y['consonant_diacritic']).type(torch.LongTensor)
        }        
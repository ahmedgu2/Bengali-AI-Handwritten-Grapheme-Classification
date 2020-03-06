import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset.test_dataset import GraphemeDataSet
from model.model_dispatcher import MODEL_DISPATCHER
import pandas as pd
import numpy as np

DEVICE = 'cuda'
TEST_BATCH_SIZE = 32
BASE_MODEL = 'resnet34'


def predict(fold):
    
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.load_state_dict(torch.load("model/checkpoints/resnet34-4-1.7213.model"))
    model.to(DEVICE)
    model.eval()

    predictions = []
    for file_indx in range(4):
        file_name = "test_image_data_{}.parquet".format(file_indx)
        dataset = GraphemeDataSet("data/", file_name)
        dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE)
        
        for batch, data in enumerate(dataloader):
            image = data['image']
            image_id = data['image_id']
            image = image.to(DEVICE)

            outs = model(image)
            g, v, c = outs
            g = np.argmax(g.cpu().detach().numpy(), axis=1)
            c = np.argmax(c.cpu().detach().numpy(), axis=1)
            v = np.argmax(v.cpu().detach().numpy(), axis=1)

            for i in range(len(image_id)):
                predictions.append(("{}_grapheme_root".format(image_id[i]), g[i]))
                predictions.append(("{}_consonant_diacritic".format(image_id[i]), c[i]))
                predictions.append(("{}_vowel_diacritic".format(image_id[i]), v[i]))
    
    df = pd.DataFrame(predictions, columns=['row_id', 'target'])
    df.to_csv("submission.csv", index=False)
    return df
            

def main():
    
    predict(0)

if __name__ == "__main__":
    main()
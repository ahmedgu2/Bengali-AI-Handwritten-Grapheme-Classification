import pandas as pd 
import numpy as np
import joblib
import glob
from tqdm import tqdm

if __name__ == "__main__":

    files = glob.glob("../data/train_*.parquet")

    for f in files:
        
        df = pd.read_parquet(f)
        images_ids = df['image_id'].values
        df = df.drop('image_id', axis=1)
        images = df.values
        
        for i, img_id in tqdm(enumerate(images_ids)):
            joblib.dump(images[i, :], "../data/image_pickle/{}.pkl".format(img_id))    


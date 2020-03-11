import pandas as pd 
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

"""
    Split data into train and validation .
"""

if __name__=="__main__":

    df = pd.read_csv("../data/train.csv")
    df.loc[:, "is_train"] = 1

    df.sample(frac=1).reset_index(drop=True)

    X = df['image_id'].values
    y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.1)

    for train_indx, val_indx in msss.split(X, y):
        df.loc[val_indx, "is_train"] = 0

    df.to_csv('../data/train_valid.csv', index=False)
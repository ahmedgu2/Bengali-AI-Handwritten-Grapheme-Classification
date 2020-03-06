import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":

    df = pd.read_csv("../data/train.csv")
    df.loc[:, "kfold"] = -1

    df.sample(frac=1).reset_index(drop=True)

    X = df['image_id'].values
    y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    kf = MultilabelStratifiedKFold(n_splits = 5)

    for fold, (train_indx, val_indx) in enumerate(kf.split(X, y)):
        df.loc[val_indx, 'kfold'] = fold

    df.to_csv("../data/train_folds.csv", index=False)

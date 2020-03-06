export CUDA_VISIBLE_DEVICES=0
export TRAIN_BATCH_SIZE=32
export EPOCHS=50
export TRAIN_DATA_DIR="data/"
export TRAIN_DATA_CSV="train_folds.csv"
export BASE_MODEL='se_resnext50'

export TRAINING_FOLDS="(0, 1, 2, 3)"
export VALIDATION_FOLDS="(4,)"
python train.py

export TRAINING_FOLDS="(0, 1, 2, 4)"
export VALIDATION_FOLDS="(3,)"
python train.py

export TRAINING_FOLDS="(0, 1, 4, 3)"
export VALIDATION_FOLDS="(2,)"
python train.py

export TRAINING_FOLDS="(0, 4, 2, 3)"
export VALIDATION_FOLDS="(1,)"
python train.py

export TRAINING_FOLDS="(4, 1, 2, 3)"
export VALIDATION_FOLDS="(0,)"
python train.py
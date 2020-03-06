import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model_dispatcher import MODEL_DISPATCHER
from dataset.train_dataset import GraphemeDataSet
import os
import ast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score
import numpy as np

DEVICE = 'cuda'

TRAIN_DATA_DIR = os.environ.get('TRAIN_DATA_DIR')
TRAIN_DATA_CSV = os.environ.get('TRAIN_DATA_CSV')

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))

BASE_MODEL = os.environ.get('BASE_MODEL')


def loss_fn(outs, targets):

    loss1 = nn.CrossEntropyLoss()(outs[0], targets[0])
    loss2 = nn.CrossEntropyLoss()(outs[1], targets[1])
    loss3 = nn.CrossEntropyLoss()(outs[2], targets[2])

    return (loss1 + loss2 + loss3) / 3


def recal(targets, preds):
    g, v, c = preds

    g = np.argmax(g.cpu().detach().numpy(), axis=1)
    v = np.argmax(v.cpu().detach().numpy(), axis=1)
    c = np.argmax(c.cpu().detach().numpy(), axis=1)

    scores = []
    scores.append(recall_score(targets[0].cpu().detach().numpy(), g, average='macro'))
    scores.append(recall_score(targets[2].cpu().detach().numpy(), v, average='macro'))
    scores.append(recall_score(targets[1].cpu().detach().numpy(), c, average='macro'))

    final_score = np.average(scores, weights=[2,1,1])

    return final_score


def train(train_loader, model, optimizer, epoch):

    model.train()
    running_loss = 0.0

    #print("Training at epoch {}".format(epoch))

    for batch, data in tqdm(enumerate(train_loader), total = len(train_loader)):

        images = data['image']
        graphme_root = data['grapheme_root'] 
        vowel_diacritic = data['vowel_diacritic']
        consonant_diacritic = data['consonant_diacritic']
        
        images = images.to(DEVICE)
        graphme_root = graphme_root.to(DEVICE)
        vowel_diacritic = vowel_diacritic.to(DEVICE)
        consonant_diacritic = consonant_diacritic.to(DEVICE)

        optimizer.zero_grad()

        outs = model(images)
        targets = (graphme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outs, targets)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

       # if batch % 10:
        #    print("batch {}/{} : {:.4f}".format(batch, len(train_loader), loss.item()))

    print("Train loss {} : {:.4f}".format(epoch, running_loss / len(train_loader)))


def evaluate(valid_loader, model, epoch):
    
    model.eval()
    running_loss = 0.0
    acc = 0.0

    #print("Validation at epoch {}".format(epoch))
    with torch.no_grad():

        for batch, data in tqdm(enumerate(valid_loader), total = len(valid_loader)):

            images = data['image']
            graphme_root = data['grapheme_root'] 
            vowel_diacritic = data['vowel_diacritic']
            consonant_diacritic = data['consonant_diacritic']

            images = images.to(DEVICE)
            graphme_root = graphme_root.to(DEVICE)
            vowel_diacritic = vowel_diacritic.to(DEVICE)
            consonant_diacritic = consonant_diacritic.to(DEVICE)

            outs = model(images)
            targets = (graphme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outs, targets)
            running_loss += loss.item()
            acc += recal(targets, outs)

        # if batch % 10:
            #    print("batch {}/{} : {:.4f}".format(batch, len(valid_loader), loss.item()))

    print("Validation loss {} : {:.4f}\n Accuracy : {:.4f}".format(epoch, running_loss / len(valid_loader), acc / len(valid_loader)))
    
    return running_loss / len(valid_loader)


def main():

    train_dataset = GraphemeDataSet(TRAIN_DATA_DIR, TRAIN_DATA_CSV, TRAINING_FOLDS)
    valid_dataset = GraphemeDataSet(TRAIN_DATA_DIR, TRAIN_DATA_CSV, VALIDATION_FOLDS)

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        num_workers = 4
    )

    valid_loader = DataLoader(
        dataset = valid_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        num_workers = 4
    )

    model = MODEL_DISPATCHER[BASE_MODEL](pretrained = True)
    model.load_state_dict(torch.load("model/checkpoints/resnet34-4-1.7213.model"))
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5, verbose=True)

    for e in range(EPOCHS):
        print("Epoch {} : ".format(e))
        #train(train_loader, model, optimizer, e)
        val_score = evaluate(valid_loader, model, e)
        scheduler.step(val_score)
        torch.save(model.state_dict(), "model/checkpoints/{}-{}-{:.4f}.model".format(BASE_MODEL, VALIDATION_FOLDS[0], val_score))

if __name__ == "__main__":
    main()
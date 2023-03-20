'''Helper file to train/evaluate models.'''
import time
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler


# Used for cutmix, see https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L279
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, use_cutmix=False):
    since = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    tr_loss = []
    tr_acc = []
    val_loss = []
    val_acc = []

    # Cutmix parameters
    beta = 1.
    cutmix_prob = 0.5

    model = model.to(device)
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            print("Phase: ", phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Cutmix see https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L228
                    if phase == 'train' and use_cutmix and np.random.rand(1) < cutmix_prob:
                        # generate mixed sample
                        lam = np.random.beta(beta, beta)
                        rand_index = torch.randperm(inputs.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                        # compute output
                        outputs = model(inputs)
                        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
            
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                tr_loss.append(epoch_loss)
                tr_acc.append(epoch_acc.item())
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.item())

            print(f'{phase} loss: {epoch_loss:.4f}; acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, [tr_loss, tr_acc, val_loss, val_acc]


def compute_metrics_on_test_set(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return confusion, precision, recall, fscore, support


def evaluate_model_on_test_set(model, data_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            all_labels.append(labels)
            # Calculate outputs by running images through the network
            outputs = model(images).to(device)
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1

    flat_preds = [item.cpu() for sub_list in all_predictions for item in sub_list]
    flat_labels = [item.cpu() for sub_list in all_labels for item in sub_list]

    test_accuracy = 100 * correct / total

    per_class_acc = {}
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        per_class_acc[classname] = 100 * float(correct_count) / total_pred[classname]

    return flat_labels, flat_preds, test_accuracy, per_class_acc


def save_training_metrics(filename, tr_loss, tr_acc, te_loss, te_acc):
    np.savez(f"{filename}.npz", tr_loss, tr_acc, te_loss, te_acc)


def load_training_metrics(filename):
    npzfile = np.load(filename)
    return npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'], npzfile['arr_3']

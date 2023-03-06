'''Helper file to train/evaluate models.'''
import time
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    tr_loss = []
    tr_acc = []
    val_loss = []
    val_acc = []

    model = model.to(device)
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
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
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

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

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

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


def evaluate_model(model, data_loader, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    correct = 0
    total = 0
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # Calculate outputs by running images through the network
            outputs = model(images).to(device)
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total

    correct_pred = {classname: 0 for classname in labels}
    total_pred = {classname: 0 for classname in labels}
    
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            all_labels.append(labels)
            outputs = model(images).to(device)
            _, predictions = torch.max(outputs, 1)
            all_predictions.append(predictions)

            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[labels[label]] += 1
                total_pred[labels[label]] += 1

    flat_preds = [item.cpu() for sub_list in all_predictions for item in sub_list]
    flat_labels = [item.cpu() for sub_list in all_labels for item in sub_list]

    confusion = confusion_matrix(flat_labels, flat_preds)

    per_class_acc = {}
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        per_class_acc[classname] = 100 * float(correct_count) / total_pred[classname]

    return test_accuracy, per_class_acc, confusion\


def save_training_metrics(filename, tr_loss, tr_acc, te_loss, te_acc):
    np.savez(f"{filename}.npz", tr_loss, tr_acc, te_loss, te_acc)


def load_training_metrics_from_npz(filename):
    npzfile = np.load(filename)
    return npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'], npzfile['arr_3']


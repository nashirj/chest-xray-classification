import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

def get_baseline_transforms(mean, std):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms

def load_xray_data(data_transforms=None):
    if data_transforms is None:
        data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ]),
    }

    data_dir = 'data/chest_xray'
    initial_train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    class_names = initial_train_set.classes
    initial_val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['val'])

    # Redistribute training/validation samples to increase the size of the validation set
    # such that 80% of the data is used for training and 20% for validation
    all_train_data = torch.utils.data.ConcatDataset([initial_train_set, initial_val_set])
    train_size = int(0.8 * len(all_train_data))
    val_size = len(all_train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(all_train_data, [train_size, val_size])

    image_datasets = {'train': train_data, 'val': val_data, 'test': test_set}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes, class_names

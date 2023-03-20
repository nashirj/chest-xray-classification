import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import WeightedRandomSampler

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
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms


def default_transform():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
    }

# TODO(): fix this function, look at kaggle notebooks for reference

def load_oversampled_xray_data(data_transforms=None):
    # Create a balanced dataset by oversampling
    # the minority classes
    if data_transforms is None:
        data_transforms = default_transform()

    data_dir = 'data/chest_xray'
    initial_train_set = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), data_transforms['train'])
    class_names = initial_train_set.classes
    initial_val_set = datasets.ImageFolder(
        os.path.join(data_dir, 'val'), data_transforms['val'])
    test_set = datasets.ImageFolder(
        os.path.join(data_dir, 'test'), data_transforms['val'])

    # Redistribute training/validation samples to increase the size of the validation set
    # such that 80% of the data is used for training and 20% for validation
    all_train_data = torch.utils.data.ConcatDataset([initial_train_set, initial_val_set])
    train_size = int(0.8 * len(all_train_data))
    val_size = len(all_train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(all_train_data, [train_size, val_size])

    # Oversample the minority classes in the training set
    # to create a balanced dataset
    train_labels = [s[1] for s in train_data]
    _, counts = torch.unique(torch.tensor(train_labels), return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    weight = [class_weights[t] for t in train_labels]
    sampler = WeightedRandomSampler(weight, len(train_labels))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=4, num_workers=4, sampler=sampler)
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=4, shuffle=True, num_workers=4)
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    dataset_sizes = {'train': len(train_data), 'val': len(val_data), 'test': len(test_set)}

    return dataloaders, dataset_sizes, class_names


def load_xray_data(data_transforms=None):
    if data_transforms is None:
        data_transforms = default_transform()

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
    train_set, val_set = torch.utils.data.random_split(all_train_data, [train_size, val_size])

    image_datasets = {'train': train_set, 'val': val_set, 'test': test_set}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes, class_names

def get_labels(dataloader):
    labels = []
    for _, label in dataloader:
        labels.append(label)
    # Flatten list of lists
    labels = [item for sublist in labels for item in sublist]
    return labels

def compute_class_weights(labels):
    # Count number of samples of each class
    num_normal = labels.count(0)
    num_pneumonia = labels.count(1)

    weight_for_0 = num_pneumonia / (num_normal + num_pneumonia)
    weight_for_1 = num_normal / (num_normal + num_pneumonia)
    class_weights = [weight_for_0, weight_for_1]
    return torch.FloatTensor(class_weights)


if __name__ == '__main__':
    dataloaders, dataset_sizes, class_names = load_oversampled_xray_data()
    print('Train size: ', dataset_sizes['train'])
    print('Val size: ', dataset_sizes['val'])
    print('Test size: ', dataset_sizes['test'])
    print('Class names: ', class_names)

    # Get number of samples of each class in train dataloader
    train_labels = [s[1] for s in dataloaders['train'].dataset]
    _, counts = torch.unique(torch.tensor(train_labels), return_counts=True)
    print('Train class counts: ', counts)

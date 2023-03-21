'''Helper file for visualizing the results of the model.'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision import transforms

def imshow(inp, mean, std, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated


def show_misclassified_images(misclassified_images, class_names, mean, std, title, save_dir="plots"):
    plt.subplot(2, 2, 1)
    i = 1
    plt.suptitle(title)
    for im, label, pred in misclassified_images:
        plt.subplot(2, 2, i)
        im = transforms.Resize((224, 224))(im)
        plt.subplot(2, 2, i)
        plt.title(f"Pred: {class_names[pred][0]}, Label: {class_names[label][0]}")
        imshow(im.cpu(), mean, std)
        plt.axis('off')
        i += 1
    if save_dir:
        title = title.lower().replace(" ", "_")
        plt.savefig(f"{save_dir}/{title}.png")
    plt.show()


def visualize_model(model, dataloaders, class_names, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def plot_confusion_matrix(cm, title, labels, save_dir="plots"):
    '''Plot confusion matrix'''
    df_cm = pd.DataFrame(cm, index = labels, columns = labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, xticklabels=True, yticklabels=True, annot=True, fmt='g', cmap='Blues')
    plt.title(f"{title} Confusion Matrix on test data")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    title = title.lower().replace(" ", "_")
    if save_dir:
        plt.savefig(f"{save_dir}/{title}_cm.png")
    plt.show()


def show_test_summary_metrics(test_accuracy, per_class_acc, cm, precision, recall, fscore, title, class_names):
    sorted_by_acc = dict(sorted(per_class_acc.items(), key=lambda item: item[1]))
    for classname, accuracy in sorted_by_acc.items():
        print(f'Accuracy for class: {classname} is {accuracy:.1f} %')

    print(f"Overall accuracy ({title}): {test_accuracy:.1f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {fscore:.3f}")

    plot_confusion_matrix(cm, title, class_names)


def plot_training_metrics(trl, tra, tel, tea, title, save_dir="plots"):
    n = [i for i in range(len(trl))]

    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(1, 2, 1)

    plt.plot(n, trl, label='train')
    plt.plot(n, tel, label='validation')
    plt.title(f"Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(n, tra, label='train')
    plt.plot(n, tea, label='validation')
    plt.title(f"Accuracy; best: {max(tea):.3f}")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()

    title = title.lower().replace(" ", "_")
    if save_dir:
        plt.savefig(f"{save_dir}/{title}_train_metrics.png")
    plt.show()

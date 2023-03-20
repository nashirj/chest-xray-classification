'''Helper file for visualizing the results of the model.'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

def imshow(inp, mean, std, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


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


def plot_confusion_matrix(cm, title, labels):
    '''Plot confusion matrix'''
    df_cm = pd.DataFrame(cm, index = labels, columns = labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, xticklabels=True, yticklabels=True, annot=True, fmt='g', cmap='Blues')
    plt.title(title + " Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
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


def plot_training_metrics(trl, tra, tel, tea, title):    
    n = [i for i in range(len(trl))]
    plt.plot(n, trl, label='train')
    plt.plot(n, tel, label='validation')
    plt.title("Training and Validation Loss, " + title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    plt.plot(n, tra, label='train acc')
    plt.plot(n, tea, label='validation acc')
    plt.title("Training and Validation Accuracy, " + title + f", best val acc: {max(tea):.3f}")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

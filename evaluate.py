import torch
import torchvision

# Iterate over test dataset and save some misclassified images
def get_misclassified_images(model, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    misclassified_images = []
    for data in dataset:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for pred, label, image in zip(preds, labels, images):
            if pred != label:
                misclassified_images.append((image, label, pred))
        # Stop if we have 4 misclassified images
        if len(misclassified_images) >= 4:
            break
    return misclassified_images

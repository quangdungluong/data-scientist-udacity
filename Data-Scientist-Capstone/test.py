import argparse
import itertools
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import orjson
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset_test import ChestXRayDatasetTest
from src.model import ChestXRayModel


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")


def test_evaluation(params):
    model = ChestXRayModel(num_classes=params['num_classes'])
    model.load_state_dict(torch.load(params['model_path']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    dataset_size, dataloader = ChestXRayDatasetTest(
        params["data_dir"], params["image_size"]).setup_data(params["batch_size"])
    y_test, y_pred = [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = outputs.max(1)

        y_test.append(labels.data.cpu().numpy())
        y_pred.append(predictions.data.cpu().numpy())

    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)

    confusion_mtx = confusion_matrix(y_test, y_pred)
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Pneunomia', 'Tuberculosis']
    plot_confusion_matrix(confusion_mtx, classes)
    report = classification_report(y_test, y_pred, digits=4)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-D", type=str)
    parser.add_argument("--model_path", "-M", type=str)
    args = parser.parse_args()

    CONFIG_PATH = "./config/train_config.json"
    params = orjson.loads(open(CONFIG_PATH, "rb").read())
    params["data_dir"] = args.data_dir
    params["model_path"] = args.model_path
    pprint(params)

    test_evaluation(params)

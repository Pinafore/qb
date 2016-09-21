import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(title, true_labels, predicted_labels, normalized=True):
    labels = list(set(true_labels) | set(predicted_labels))

    if normalized:
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    ax = plt.gca()
    ax.grid(False)

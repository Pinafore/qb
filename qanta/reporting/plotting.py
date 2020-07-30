import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(title, true_labels, predicted_labels, normalized=True):
    labels = list(set(true_labels) | set(predicted_labels))

    if normalized:
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.grid(False)
    return fig, ax

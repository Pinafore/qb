from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pickle
from qanta.util.constants import REPRESENTATION_RES_TARGET, REPRESENTATION_TSNE_TARGET
from sklearn.manifold import TSNE


def plot_representations():
    with open(REPRESENTATION_RES_TARGET, 'rb') as f:
        data = pickle.load(f)
    split_index = len(data[0])
    data = list(chain.from_iterable(np.squeeze(d) for d in data))
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embeds = tsne.fit_transform(data)
    for i, (x, y) in enumerate(low_dim_embeds):
        plt.scatter(x, y, color=('red' if i < split_index else 'blue'))
    plt.savefig(REPRESENTATION_TSNE_TARGET)


if __name__ == '__main__':
    plot_representations()

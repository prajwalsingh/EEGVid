import os
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import style

style.use('ggplot')

class K_means:
    def __init__(self, n_clusters=40, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def transform(self, embed, gt_labels):
        pred_labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(embed)
        score       = self.cluster_acc(gt_labels, pred_labels)
        return score

    # Thanks to: https://github.com/k-han/DTC/blob/master/utils/util.py
    def cluster_acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size


class TsnePlot:
    def __init__(self, perplexity=30, learning_rate=200, n_iter=1000):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
    def plot(self, embedding, labels, score, train_type, test_type, save_path, file_name, data_type, cluster_names):
        os.makedirs(save_path, exist_ok=True)

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val) / (max_val - min_val)

        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, len(unique_labels)))

        plt.figure(figsize=(3, 4))
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', labelsize=11)

        # Store cluster center positions for annotation
        cluster_centers = {}

        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_points = reduced_embedding[mask]

            # Compute the cluster center (mean of points)
            center_x, center_y = np.mean(cluster_points, axis=0)
            cluster_centers[label] = (center_x, center_y)

            # Scatter plot
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors[i]], label=label, alpha=0.6)

        # Annotate each cluster with its name
        for label, (center_x, center_y) in cluster_centers.items():
            ax.text(center_x, center_y, cluster_names[label], fontsize=10, fontweight='bold',
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

        # ax.legend(loc='center left', fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(1, 0.5))
        ax.legend(ncol=1)
        # plt.title('{} [t-SNE] KMeans score: {}/ Train: {}/ Test: {}'.format(data_type, score, train_type, test_type))
        plt.tight_layout()
        plt.savefig(save_path + f'/{file_name}.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        return reduced_embedding

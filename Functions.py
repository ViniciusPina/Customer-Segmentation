import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ipympl
import warnings
from pandas_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import os
warnings.filterwarnings('ignore')
def elbow_silhouette_graphic(X,random_state=42,intervalo_k=(2,11)):
    elbow = {}
    silhouette = []
    k_range = range(*intervalo_k)


    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), tight_layout=True)

    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
        kmeans.fit(X)
        elbow[i] = kmeans.inertia_
        labels = kmeans.labels_
        silhouette.append(silhouette_score(X, labels))


    sns.lineplot(x=list(elbow.keys()), y=list(elbow.values()), ax=axs[0])
    axs[0].set_xlabel('K')
    axs[0].set_ylabel('Inertia')
    axs[0].set_title('Elbow Method')


    sns.lineplot(x=list(k_range), y=silhouette, ax=axs[1])
    axs[1].set_xlabel('K')
    axs[1].set_ylabel('Silhouette Score')
    axs[1].set_title('Silhouette Score Method')
    
    plt.show()
    return
    
    
    
    
    
    
    

def view_clusters(
    dataframe,
    colunas,
    quantidade_cores,
    centroids,
    mostrar_centroids=True,
    mostrar_pontos=False,
    coluna_clusters=None,
):
    """
    Generate 3D graph with clusters

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe with the data.
    colunas: List[str]
        List with the name of the columns (strings) to be used.
    quantidade_cores: int
        Number of colors for the chart.
    centroids: np.ndarray
        Array with centroids.
    mostrar_centroids : bool, optional
        Whether the graph will show the centroids or not, by default True
    mostrar_pontos: bool, optional
        Whether the graph will show the points or not, by default False
    coluna_clusters : List[int], optional
        Column with cluster numbers to color the points (if show_points is True), by default None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    cores = plt.cm.tab10.colors[:quantidade_cores]
    cores = ListedColormap(cores)

    x = dataframe[colunas[0]]
    y = dataframe[colunas[1]]
    z = dataframe[colunas[2]]

    for i, centroid in enumerate(centroids):
        if mostrar_centroids:
            ax.scatter(*centroid, s=500, alpha=0.5)
            ax.text(
                *centroid,
                f"{i}",
                fontsize=20,
                horizontalalignment="center",
                verticalalignment="center",
            )

    if mostrar_pontos:
        s = ax.scatter(x, y, z, c=coluna_clusters, cmap=cores)
        ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 1))

    ax.set_xlabel(colunas[0])
    ax.set_ylabel(colunas[1])
    ax.set_zlabel(colunas[2])
    ax.set_title("Clusters")

    plt.show()
    return

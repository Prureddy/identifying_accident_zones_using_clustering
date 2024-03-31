import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from io import BytesIO
import base64


def load_accident_data():
    # Loading accident data
    accidents1 = pd.read_csv('data/accidents_2005_to_2007.csv', encoding='latin-1')
    accidents2 = pd.read_csv('data/accidents_2009_to_2011.csv', encoding='latin-1')
    accidents3 = pd.read_csv('data/accidents_2012_to_2014.csv', encoding='latin-1')
    total_accidents = pd.concat([accidents1, accidents2, accidents3])
    locations = total_accidents[['Longitude', 'Latitude']]
    locations = locations.dropna()
    return locations


def plot_top_clusters(locations, km_labels, top_clusters):
    # Plotting top density-based clusters found by Kmeans-DB
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(locations['Longitude'], locations['Latitude'], s=locations['counts'] / locations['weight'] * 10)
    for i in top_clusters:
        points = locations[km_labels == i]
        plt.scatter(points['Longitude'], points['Latitude'], s=points['counts'] / points['weight'] * 10)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Top density-based clusters found by Kmeans-DB')

    # Save plot as base64 encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plot_data

def load_traffic_flow_data():
    # Loading traffic flow data
    aadf = pd.read_csv('data/dft_traffic_counts_aadf.csv', encoding='latin-1')
    aadf = aadf[['longitude', 'latitude', 'all_motor_vehicles']]
    aadf = aadf.dropna()
    return aadf


def preprocess_data(locations, aadf):
    # Preprocessing data
    locations = np.round(locations, decimals=2)
    locations = locations.groupby(['Longitude', 'Latitude']).size().reset_index(name='counts')

    aadf = np.round(aadf, decimals=2)
    aadf = aadf.groupby(['longitude', 'latitude']).agg('mean').reset_index()

    return locations, aadf


def combine_data(locations, aadf):
    # Combining the data
    weights = np.full(len(locations), 0.0)
    mindists = np.full(len(locations), 10000.0)

    for i in range(len(aadf)):
        weight = aadf.loc[i][2]
        dists = np.linalg.norm(list(aadf.loc[i][:2]) - locations.values[:, 0:2], axis=1)
        weights[dists < mindists] = weight
        mindists[dists < mindists] = dists[dists < mindists]
    weights[weights == 0] = 0.5

    locations['weight'] = weights
    return locations


def run_dbscan(locations):
    # DBSCAN
    db = DBSCAN(eps=0.1, min_samples=10).fit(locations.values[:, 0:2],
                                             sample_weight=locations['counts'] / locations['weight'])
    return db.labels_


def run_kmeans(locations):
    # KMeans-DB
    km = KMeans(n_clusters=100)
    km.fit(locations[['Longitude', 'Latitude']], sample_weight=locations['counts'] / locations['weight'])
    return km.labels_


def get_top_density_clusters(locations, km_labels, n=5):
    densities = []
    for x in set(km_labels):
        points = locations[km_labels == x]
        density = sum(points['counts'] / points['weight']) / len(points)
        densities.append((density, x))

    # Sort the clusters by density in descending order
    top_clusters = sorted(densities, reverse=True)[:n]
    top_clusters_labels = [y[1] for y in top_clusters]

    return top_clusters_labels





def find_cluster_names(cluster_labels):
    # Define a dictionary to map cluster labels to cluster names
    cluster_names = {
        0: "Cluster A",
        1: "Cluster B",
        2: "Cluster C",
        # Add more mappings as needed
    }

    # Initialize a list to store the cluster names corresponding to each label
    names = []

    # Iterate through each cluster label and find its corresponding name
    for label in cluster_labels:
        if label in cluster_names:
            names.append(cluster_names[label])
        else:
            names.append("Unknown")

    return names

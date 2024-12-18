import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.arraysetops import unique
from PIL import Image
from sklearn import metrics
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler

from support_funs import rotate_bound_modified, write_result

#from support_funs import write_cluster_image,write_cluster_image_2,write_result
#get_ipython().run_line_magic('matplotlib', 'inline')

pixels_for_cluster = []
diff_perc_list = []


def save_plot(directory, image_name):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the path to save the image
    save_path = os.path.join(directory, f"{image_name}.png")
    
    # Save the plot using plt.savefig
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600)


def cluster_pixels(image, ground_truth, map_folder, run_number, name, map_size):

    plt.close()
    plt.clf()

    if map_size == 'small':
        DIFFERENCE_THRESHOLD = 12
        CLUSTER_THRESHOLD = 1.35
        print('its small')
    elif map_size == 'medium':
        DIFFERENCE_THRESHOLD = 8
        CLUSTER_THRESHOLD = 0.8
        print('its medium')
    elif map_size == 'large':
        DIFFERENCE_THRESHOLD = 2
        CLUSTER_THRESHOLD = 60
        print('its large')
    else:
        print("error switch" + map_folder)
        return

    im1 = image
    im2 = ground_truth

    if im1 is None or im2 is None:
        print("ground_truth problem ", ground_truth)
        print("cluster image problem ", image)
        return 0, 0, pixels_for_cluster
    # if im1.shape != im2.shape:
    #     print(f"Error: Image dimensions do not match for {image} and {ground_truth}")
    #     return 2, 0, None
    GT_pixels = np.where(im2 == 254)

    GT_pixels_size = GT_pixels[1].size  # if len(GT_pixels) > 1 else 0
    diff_pixels = np.where(im1 != im2)

    if len(diff_pixels) > 1:
        diff_pixels_size = diff_pixels[1].size
    else:
        print(f"No valid difference pixels found in image")
        return 0, 0, pixels_for_cluster

    difference_percentage = (diff_pixels_size / GT_pixels_size) * 100  # if GT_pixels_size > 0 else 0

    if difference_percentage > 50:
        print(f"map_folder: {map_folder}")
    diff_perc_list.append(difference_percentage)
    print("diff_pixels_size", diff_pixels_size)
    
    if difference_percentage > DIFFERENCE_THRESHOLD:
        return 0, difference_percentage, pixels_for_cluster

    if difference_percentage == 0:
        return 1, difference_percentage, pixels_for_cluster

    scale_percent = 15  # percentage of original size
    width = int(im1.shape[1] * scale_percent / 100)
    height = int(im1.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    # im1 = cv2.resize(im1, dim, interpolation = cv2.INTER_NEAREST)
    # im2 = cv2.resize(im2, dim, interpolation = cv2.INTER_NEAREST)
    GT_pixels_resized = np.where(cv2.resize(im2, dim, interpolation=cv2.INTER_LINEAR) != 205)
    GT_pixels_resized = GT_pixels_resized[1].size

    down_image = cv2.absdiff(im1, im2)
    down_image = cv2.resize(down_image, dim, interpolation=cv2.INTER_LINEAR)
    result = np.where(down_image == 49)

    if result[0].size == 0:
        return 1, difference_percentage, pixels_for_cluster

    listOfCoordinates = list(zip(result[0], result[1]))
    df = pd.DataFrame(data=listOfCoordinates, columns=["x_coord", "y_coord"])

    # setting for frontiers
    # db = DBSCAN(eps=3, min_samples=5).fit(df.values)
    # setting for clusters
    db = DBSCAN(eps=10, min_samples=10).fit(df.values)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    noisy_points = db.labels_ == -1
    cluster_points = ~noisy_points

    print("Number of clusters = %d"%n_clusters_)
    print(np.unique(db.core_sample_indices_))
    print(db.core_sample_indices_)
    print("Number of cluster points = %d"%sum(cluster_points))
    print("Number of noisy points = %d"%sum(noisy_points))

    if sum(cluster_points) == 0:
        return 1, difference_percentage, pixels_for_cluster

    number_of_points_per_cluster = []

    sup_labels = [i for i in labels if i != -1]
    for label in np.unique(sup_labels):
        number_of_points_per_cluster.append(sup_labels.count(label))
        # pixels_for_cluster.append(sup_labels.count(label))
    
    clusters_percentage=[(i/GT_pixels_resized)*100 for i in number_of_points_per_cluster]
    print(f'clusters_percentage: {clusters_percentage}')
    # old way with percentage
    # max_cluster = max(clusters_percentage)

    # new way with number of pixels
    max_cluster = max(number_of_points_per_cluster)
    pixels_for_cluster.append(max_cluster)
    print(f'number_of_points_per_cluster: {number_of_points_per_cluster}')
    print(f'pixels_for_cluster: {pixels_for_cluster}')
    print(f'max cluster: {max_cluster}')    

    if max_cluster > CLUSTER_THRESHOLD or difference_percentage > DIFFERENCE_THRESHOLD:
        print('Maximum cluster:', max_cluster)

        # # Cluster threshold
        # cluster_size_threshold = 50 

        # Frontiers threshold
        cluster_size_threshold = 20 

        # Create a new list to store colors based on cluster size
        cluster_colors = np.zeros(len(labels), dtype=int)

        # Loop over the unique cluster labels
        for label in np.unique(labels):
            if label == -1:
                # Skip noise points
                continue
    
            # Select points belonging to the current cluster
            cluster_points = df[labels == label]
    
            # Check the size of the cluster
            if len(cluster_points) > cluster_size_threshold:
                cluster_colors[labels == label] = 1  # Larger than threshold: green
            else:
                cluster_colors[labels == label] = 0  # Smaller than threshold: red


        # Plot 1: Original cluster visualization

        # plt.subplot(1, 2, 1)
        # plt.scatter(df['y_coord'], df['x_coord'], s=0.2, marker=',', c=labels, cmap=plt.get_cmap('tab20'))
        # plt.xlim([0, width])
        # plt.ylim([0, height])
        # ax1 = plt.gca()
        # ax1.set_ylim(ax1.get_ylim()[::-1])
        # ax1.set_aspect('equal')
        # plt.title('Cluster Visualization')


        # Plot 1: Original cluster visualization with color coding
        plt.subplot(1, 2, 1)

        cmap = plt.cm.colors.ListedColormap(['red', 'green'])

        # Plot the clusters using a scatter plot and the assigned color map
        plt.scatter(df['y_coord'], df['x_coord'], s=0.2, marker=',', c=cluster_colors, cmap=cmap)

        plt.xlim([0, width])
        plt.ylim([0, height])
        ax1 = plt.gca()
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_aspect('equal')
        plt.title('Cluster Visualization')

        # # This can be a good way of postprocessing
        # import networkx as nx
        # from sklearn.neighbors import NearestNeighbors

        # def separate_disconnected_areas(cluster_points, eps):
        #     """
        #     This function takes the points of a cluster and returns disconnected subclusters.
        #     cluster_points: a list or array of points (e.g., pixel coordinates in 2D).
        #     eps: the epsilon value used for DBSCAN, also used to determine connectivity.
        #     """
        #     # Build a nearest neighbors graph
        #     nbrs = NearestNeighbors(radius=eps).fit(cluster_points)
        #     adjacency_matrix = nbrs.radius_neighbors_graph(cluster_points).toarray()
    
        #     # Create a graph from the adjacency matrix
        #     G = nx.Graph(adjacency_matrix)
    
        #     # Find connected components (subclusters)
        #     subclusters = [list(component) for component in nx.connected_components(G)]
    
        #     return subclusters

        # Plot 2: Overlay clusters on the reference image (unchanged)
        plt.subplot(1, 2, 2)

        # Resize image to proper dimension
        resized_reference = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

        # Create a mask for the clusters
        cluster_mask = np.zeros_like(resized_reference, dtype=np.uint8)
        for label in np.unique(labels):
            if label != -1:
                cluster_points = df[labels == label]
                for index, point in cluster_points.iterrows():
                    cluster_mask[int(point['x_coord']), int(point['y_coord'])] = [255, 0, 0]  # Red color for clusters

        # Resize the mask to match the reference image
        resized_cluster_mask = cv2.resize(cluster_mask, dim, interpolation=cv2.INTER_NEAREST)

        # Overlay the cluster mask on the reference image with transparency
        overlay = cv2.addWeighted(resized_reference, 0.6, resized_cluster_mask, 0.4, 0)

        plt.imshow(overlay)
        plt.title('Reference Image with Cluster Overlay')

        # plt.show()
        saved_dir = '/home/michele/Documents/StakanovThesis/robot-aware-exploration/Results/Saved_Frontier_classification/'
        print(f"map:   {name}")
        save_plot(saved_dir, f"classified_{name}")

        return 0, difference_percentage, pixels_for_cluster
    
    print('Clustering of map', image, 'completed\n')
    return 1, difference_percentage, pixels_for_cluster
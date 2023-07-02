import pandas as pd
import numpy as np
import sys
import os
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from collections import Counter

##from utils.focal import FOCAL
from utils.Adjusted_FOCAL import FOCAL
from utils.Extract import extract_AP


def cover_space_sample(xyz, radius, max_samples):
    """
    B - Batch, N, npoint - number of points, d-dim coordinates
    Input:
        xyz: pointcloud data, [N, d]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, d]
    """
    N, d = xyz.shape
    grouped = np.zeros(N)
    radius = radius**2

    centroids = np.zeros((max_samples, d), dtype = np.float)
    centroids_idx = np.zeros(max_samples, dtype = np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N, dtype = np.int64)

    for i in range(max_samples):
        # Sample centroid and calculate all the other points' distance to it
        centroids_idx[i] = farthest
        centroids[i] = xyz[farthest, :]
        dist = np.sum((xyz - centroids[i])**2, -1)

        # Find next centroid to sample
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

        # Check space covering
        mask = dist < radius
        grouped[mask] = 1
        if grouped.all():
            return centroids[:i], centroids_idx[:i]

    print("Not all spaces were sampled")

    return centroids, centroids_idx


def query_ball_point(radius, xyz, new_xyz):
    """

    Input:
        radius: local region radius
        xyz: all points, [N, d]
        new_xyz: query points, [M, d]
    Return:
        group_idx: grouped points index, [M, nsample]
    """

    # Sampling by nearest neighbors to center
    sqrdists = distance_matrix(new_xyz, xyz, p=2)
    sqrdists[sqrdists > radius] = np.nan
    group_idx = np.argsort(sqrdists, axis=1)
    sqrdists = np.take_along_axis(sqrdists, group_idx, axis=1)

    mask = (sqrdists == sqrdists) # mask nan
    nsample = mask.sum(axis=1).max()

    group_idx = group_idx[:, :nsample]
    mask = mask[:, :nsample]

    return group_idx, mask


def sample_and_group(xyz, radius, max_samples=100, min_npoints=-1):
    """

    Input:
        xyz: input points position data, [N, d]
        radius: local region radius
        max_samples: ?
        min_npoints: min number of points in group (filter groups with less points)
    Return:
        new_xyz: sampled points position data, [nsample, 3]
        new_points: sampled points data, [nsample, npoint, 3]
    """
    centroids, centroids_idx = cover_space_sample(xyz, radius, max_samples)

    group_idx, mask = query_ball_point(radius, xyz, centroids)
    rows = []
    groups = []
    for row in range(group_idx.shape[0]):
        if mask[row].sum() > min_npoints:
            rows.append(row)
            groups.append(group_idx[row, mask[row]])

    return centroids[rows], groups

'''
def dbscan_cluster_and_group(xyz,
                             min_npoints = 15,
                             eps = 15,
                             min_cluster_points = 15, 
                             max_std_distance = 2.5):
    """Use DBSCAN to cluster a given list of points, then bound them by rects.

        :param xyz: list of lists where each inner list represents a point (row_x, col_y).
        :param eps: int, DBSCAN parameter maximum distance (in pixels) between two points to consider
                            them as a same cluster
        :param min_samples: int, DBSCAN parameter minimum number of adjacent points in the cluster to define
                        a point as core point to cluster.
        :param min_cluster_points: int, minimum number of points to be considered as cluster.
        :param max_std_distance: filter out points in cluster that exceeds a factor of std

       :return:
            new_xyz: sampled points position data, [nsample, 3]
            new_points: sampled points data, [nsample, npoint, 3]
    """

    try:
        clustering = DBSCAN(eps = eps, min_samples = min_npoints).fit(xyz)
        labels = clustering.labels_
        print(type(clustering))
        print('DBSCAN Clustering: ', clustering)
        
    except BaseException as be:
        print(be)
                 
    print('DBSCAN Clustering: ', clustering)
    labels = clustering.labels_
    xyz['Label'] = labels

    return xyz

def hdbscan_cluster_and_group(xyz,
                              min_cluster_points = 15,
                              epsilon_threshold = -9999,
                              min_samples = 1,
                              extracting_alg = "leaf",
                              alpha = 1.0,
                              max_std_distance = 2.5,):
    """Use DBSCAN to cluster a given list of points, then bound them by rects.

        :param xyz: list of lists where each inner list represents a point (row_x, col_y).
        :param min_samples: int,
        :param epsilon_threshold: int,
        :param extracting_alg: str,
        :param alpha: int,
        :param min_cluster_points: int, minimum number of points to be considered as cluster.
        :param max_std_distance: filter out points in cluster that exceeds a factor of std

       :return:
            new_xyz: sampled points position data, [nsample, 3]
            new_points: sampled points data, [nsample, npoint, 3]
    """

    try:
        if epsilon_threshold != -9999:
            print("Using epsilon threshold: %d" % epsilon_threshold)
            clustering = hdbscan.HDBSCAN(min_cluster_size = min_cluster_points, 
                                         min_samples = min_samples,
                                         alpha = alpha,
                                         cluster_selection_method = extracting_alg,
                                         cluster_selection_epsilon = epsilon_threshold)
        else: 
            clustering = hdbscan.HDBSCAN(min_cluster_size = min_cluster_points, 
                                         min_samples = min_samples,
                                         alpha = alpha,
                                         cluster_selection_method = extracting_alg)
        clustering.fit(xyz)
    except BaseException as be:
        print(be)

    print('HDBSCAN Clustering: ', clustering)
    labels = clustering.labels_
    xyz['Label'] = labels
    
    return xyz
'''

def general_cluster_and_group(xyz, labels, fname):
    """
    @param: xyz - dataframe, x,y,z,labels
    """

    unique_labels = set(labels)
    cluster_num = len(unique_labels) - (1 if -1 in labels else 0)
    cluster_props_dict = {}

    clusters = dict()
    for label in unique_labels:
        if label != -1:
            ln = xyz['Label'].value_counts()[label]
            if ln < 10:
                xyz.loc[xyz['Label'] == label, 'Label'] = -1
            else:
                clusters[str(int(label))] = []

    clustered_df = xyz.loc[xyz['Label'] != -1]
    if len(clustered_df.index) > 0:
        
        clustered_lst = clustered_df.values.tolist()
        for p in clustered_lst:
            l = p[3]
            clusters[str(int(l))].append(p)
        sorted_vals = sorted(clusters.values(), reverse = True)
        sorted_clusters = dict()
        for i in sorted_vals:
            for k in clusters.keys():
                if clusters[k] == i:
                    sorted_clusters[k] = clusters[k]
                    break
        number_of_locs = len(xyz.index)
        number_of_locs_assigned_to_clusters = len(clustered_df.index)

        img_props, clstr_props, cluster_props_dict = extract_AP(clusters)
        
        # Inserting columns to the beginning of the DataFrames
        clstr_props.insert(loc = 0,
                           column = 'File Name',
                           value = [fname]*len(clstr_props.index))
        img_props.insert(loc = 0,
                         column = 'File Name',
                         value = [fname])
        img_props.insert(loc = 1,
                         column = 'Number of Clusters',
                         value = [cluster_num])
        img_props.insert(loc = 2,
                         column = 'Total Number of Localisations',
                         value = [number_of_locs])
        img_props.insert(loc = 3,
                         column = 'Number of Localisations Assigned to Clusters',
                         value = [number_of_locs_assigned_to_clusters])

    else:
        img_props = pd.DataFrame()
        clstr_props = pd.DataFrame()
    
    return img_props, clstr_props, cluster_props_dict


def dbscan_cluster_and_group(xyz,
                             min_npoints = 15,
                             eps = 15,
                             min_cluster_points = 15, 
                             max_std_distance = 2.5,
                             metric = 'euclidean',
                             alg = 'auto',
                             fname = ''):
    """Use DBSCAN to cluster a given list of points, then bound them by rects.

        :param xyz: list of lists where each inner list represents a point (row_x, col_y).
        :param eps: int, DBSCAN parameter maximum distance (in pixels) between two points to consider
                            them as a same cluster
        :param min_samples: int, DBSCAN parameter minimum number of adjacent points in the cluster to define
                        a point as core point to cluster.
        :param min_cluster_points: int, minimum number of points to be considered as cluster.
        :param max_std_distance: filter out points in cluster that exceeds a factor of std

       :return:
            new_xyz: sampled points position data, [nsample, 3]
            new_points: sampled points data, [nsample, npoint, 3]
    """

    try:
        clustering = DBSCAN(eps = eps, min_samples = min_npoints,
                            metric = metric, algorithm = alg, leaf_size = 30)

        clustering.fit(xyz)
        
    except BaseException as be:
        print(be)
                 
    print(clustering)
    labels = clustering.labels_
    xyz['Label'] = labels
    
    img_props, clstr_props, cluster_props_dict = general_cluster_and_group(xyz, labels, fname)

    if len(img_props.index) > 0:

        img_props['Scan Parameters'] = [['Algorithm: DBSCAN',
                                         'min_npoints: ' + str(min_npoints), 'epsilon: ' + str(eps),
                                         'min_cluster_points: ' + str(min_cluster_points),
                                         'max_std_distance: ' + str(max_std_distance),
                                         'metric: ' + str(metric),
                                         'alg: ' + str(alg)]]
    

    return img_props, clstr_props, cluster_props_dict, xyz


def hdbscan_cluster_and_group(xyz,
                              min_cluster_points = 15,
                              epsilon_threshold = -9999,
                              min_samples = 1,
                              extracting_alg = "leaf",
                              alpha = 1.0,
                              max_std_distance = 2.5,
                              fname = ''):
    """Use DBSCAN to cluster a given list of points, then bound them by rects.

        :param xyz: list of lists where each inner list represents a point (row_x, col_y).
        :param min_samples: int,
        :param epsilon_threshold: int,
        :param extracting_alg: str,
        :param alpha: int,
        :param min_cluster_points: int, minimum number of points to be considered as cluster.
        :param max_std_distance: filter out points in cluster that exceeds a factor of std

       :return:
            new_xyz: sampled points position data, [nsample, 3]
            new_points: sampled points data, [nsample, npoint, 3]
    """

    try:
        if epsilon_threshold != -9999:
            print("Using epsilon threshold: %d" % epsilon_threshold)
            clustering = hdbscan.HDBSCAN(min_cluster_size = min_cluster_points, 
                                         min_samples = min_samples,
                                         alpha = alpha,
                                         cluster_selection_method = extracting_alg,
                                         cluster_selection_epsilon = epsilon_threshold)
        else: 
            clustering = hdbscan.HDBSCAN(min_cluster_size = min_cluster_points, 
                                         min_samples = min_samples,
                                         alpha = alpha,
                                         cluster_selection_method = extracting_alg)

        clustering.fit(xyz)

    except BaseException as be:
        print(be)

    print(clustering)
    labels = clustering.labels_
    xyz['Label'] = labels
    img_props, clstr_props, cluster_props_dict = general_cluster_and_group(xyz, labels, fname)

    if len(img_props.index) > 0:
        img_props['Scan Parameters'] = [['Algorithm: HDBSCAN',
                                         'min_cluster_points: ' + str(min_cluster_points),
                                         'epsilon_threshold: ' + str(epsilon_threshold),
                                         'min_samples: ' + str(min_samples),
                                         'extracting_alg: ' + str(extracting_alg),
                                         'alpha: ' + str(alpha),
                                         'max_std_distance: ' + str(max_std_distance)]]

    return img_props, clstr_props, cluster_props_dict, xyz




def focal_cluster_and_group(xyz,
                            sigma = 55,
                            minL = 1,
                            minC = 25,
                            minPC = 0,
                            fname = ''):
    """Use FOCAL to cluster a given list of points, then bound them by rects.

        :param xyz: list of lists, each inner list represents a point (row_x, col_y).
        :param sigma: int, grid size
        :param minL: int, density threshold
        :param minC: int, cluster size threshold
        :param max_std_distance: filter out points in cluster that exceeds a factor of std
        :param minPC: filter out clusters with an average photon-count of less than minPC

       :return:
            new_xyz: sampled points position data, [nsample, 3]
            new_points: sampled points data, [nsample, npoint, 3]
    """
    print('minPC in focal_cluster_and_group = ', minPC)
    all_locs_df, clustered_df = FOCAL(xyz, minL, minC, sigma, minPC)

    labels = all_locs_df['Label']
    img_props, clstr_props, cluster_props_dict = general_cluster_and_group(all_locs_df, labels, fname)

    if len(img_props.index) > 0:
        img_props['Scan Parameters'] = [['Algorithm: FOCAL',
                                         'Sigma: ' + str(sigma),
                                         'minL: ' + str(minL),
                                         'minC: ' + str(minC),
                                         'minPC: ' + str(minPC)]]

    return img_props, clstr_props, cluster_props_dict, all_locs_df
    

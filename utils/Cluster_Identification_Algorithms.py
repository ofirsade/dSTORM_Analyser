import pandas as pd
import numpy as np
import sys
import os
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from collections import Counter

from utils.Adjusted_FOCAL import FOCAL
##from utils.Extract import extract_AP
from utils.Extract import extract_AP, pca_outliers_and_axes


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


def general_cluster_and_group(xyz, labels, fname, d2_th, d3_th):
    """
    @params:
    @@ xyz - dataframe: x,y,z,labels
    @@ labels - list: unique cluster labels
    @@ fname - str: current file name
    @@ d2_th - float: 2D density threshold
    @@ d3_th - float: 3D density threshold
    @returns:
    @@ img_pros - dataframe: properties of the entire scan
    @@ clstr_props - dataframe: properties of each cluster in the scan
    @@ noise - list: points that were discarded due to density threshold to add to non-clustered localisations dataframe
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

        img_props, clstr_props, cluster_props_dict, dropped_labels = extract_AP(clusters, d2_th, d3_th)

        # Remove clusters that were eliminated due to low density
        for label in dropped_labels:
            ln = xyz.loc[xyz['Label'] == label, 'Label'] = -1
            del clusters[str(label)]

        sorted_vals = sorted(clusters.values(), reverse = True)
        sorted_clusters = dict()
        for i in sorted_vals:
            for k in clusters.keys():
                if clusters[k] == i:
                    sorted_clusters[k] = clusters[k]
                    break
        number_of_locs = len(xyz.index)
        number_of_locs_assigned_to_clusters = len(clustered_df.index)
        
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
                         column = 'Total Number of Analysed Localisations',
                         value = [number_of_locs])
        img_props.insert(loc = 3,
                         column = 'Number of Localisations Assigned to Clusters',
                         value = [number_of_locs_assigned_to_clusters])

    else:
        img_props = pd.DataFrame()
        clstr_props = pd.DataFrame()
        noise = []
    
    return img_props, clstr_props, cluster_props_dict, xyz


def dbscan_cluster_and_group(xyz,
                             eps = 15,
                             min_samples = 15, 
                             max_std_distance = 2.5,
                             metric = 'euclidean',
                             alg = 'auto',
                             fname = '',
                             pca_stddev = None,
                             d2_th = 0.0,
                             d3_th = 0.0):
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
        clustering = DBSCAN(eps = eps, min_samples = min_samples,
                            metric = metric, algorithm = alg, leaf_size = 30)

        clustering.fit(xyz)
        
    except BaseException as be:
        print(be)
                 
    print(clustering)
    labels = clustering.labels_
    print('DBSCAN Labels:\n', labels)
    xyz['Label'] = labels
    ## Changed Code from here until line 217
    orig_noise = xyz.loc[xyz['Label'] == -1]
    xyzl = xyz.copy()
    if pca_stddev != None:
        data = pd.DataFrame()
        for l in set(labels):
            if l != -1:
                cluster_df = xyz.loc[xyz['Label'] == l]
                cluster = cluster_df[['x', 'y', 'z']].to_numpy()
                denoised_cluster, dropped_noise = pca_outliers_and_axes(cluster, pca_stddev)
                denoised_cluster_df = pd.DataFrame(denoised_cluster, columns = ['x', 'y', 'z'])
                dropped_noise_df = pd.DataFrame(dropped_noise, columns = ['x', 'y', 'z'])
                denoised_cluster_df['Label'] = [l]*len(denoised_cluster_df.index)
                dropped_noise_df['Label'] = [-1]*len(dropped_noise_df.index)
                data = pd.concat([data, denoised_cluster_df, dropped_noise_df], axis = 0)
        xyz_pca = pd.concat([data, orig_noise])
        ul = set(xyz_pca['Label'].values.tolist())
        if -1 in ul:
            ul.remove(-1)
        if len(ul) > 0:
            ls = set(xyz_pca.loc[xyz_pca['Label'] != -1])
            if len(ls) > 0:
                img_props, clstr_props, cluster_props_dict, xyzl = general_cluster_and_group(xyz_pca, xyz_pca['Label'].values.tolist(), fname, d2_th, d3_th)
                
                if len(img_props.index) > 0:

                    img_props['Scan Parameters'] = [['Algorithm: DBSCAN',
                                                     'epsilon: ' + str(eps),
                                                     'min_samples: ' + str(min_samples),
                                                     'max_std_distance: ' + str(max_std_distance),
                                                     'metric: ' + str(metric),
                                                     'alg: ' + str(alg),
                                                     'PCA: ' + str(pca_stddev)]]
##                for p in noise:
##                    row_num = xyz.loc[(xyz['x'] == p[0]) & (xyz['y'] == p[1]) & (xyz['z'] == p[2])].index.tolist()
##                    for n in row_num:
##                        xyz.at[n, 'Label'] = -1
                
            else:
                img_props = pd.DataFrame()
                clstr_props = pd.DataFrame()
                cluster_props_dict = {}
        
        else:
            img_props = pd.DataFrame()
            clstr_props = pd.DataFrame()
            cluster_props_dict = {}
    
    else:
        img_props, clstr_props, cluster_props_dict, xyzl = general_cluster_and_group(xyz, labels, fname, d2_th, d3_th)

        if len(img_props.index) > 0:

            img_props['Scan Parameters'] = [['Algorithm: DBSCAN',
                                             'epsilon: ' + str(eps),
                                             'min_samples: ' + str(min_samples),
                                             'max_std_distance: ' + str(max_std_distance),
                                             'metric: ' + str(metric),
                                             'alg: ' + str(alg),
                                             'PCA: No PCA']]
##        for p in noise:
##            row_num = xyz.loc[(xyz['x'] == p[0]) & (xyz['y'] == p[1]) & (xyz['z'] == p[2])].index.tolist()
##            for n in row_num:
##                xyz.at[n, 'Label'] = -1
    
    return img_props, clstr_props, cluster_props_dict, xyzl


def hdbscan_cluster_and_group(xyz,
                              min_cluster_points = 15,
                              epsilon_threshold = -9999,
                              min_samples = 1,
                              extracting_alg = "leaf",
                              alpha = 1.0,
                              max_std_distance = 2.5,
                              fname = '',
                              pca_stddev = None,
                              d2_th = 0.0,
                              d3_th = 0.0):
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
    orig_noise = xyz.loc[xyz['Label'] == -1]
    xyzl = xyz.copy()
    try:
        if pca_stddev != None:
            data = pd.DataFrame()
            for l in set(labels):
                if l != -1:
                    cluster_df = xyz.loc[xyz['Label'] == l]
                    cluster = cluster_df[['x', 'y', 'z']].to_numpy()
                    denoised_cluster, dropped_noise = pca_outliers_and_axes(cluster, pca_stddev)
                    denoised_cluster_df = pd.DataFrame(denoised_cluster, columns = ['x', 'y', 'z'])
                    dropped_noise_df = pd.DataFrame(dropped_noise, columns = ['x', 'y', 'z'])
                    denoised_cluster_df['Label'] = [l]*len(denoised_cluster_df.index)
                    dropped_noise_df['Label'] = [-1]*len(dropped_noise_df.index)
                    data = pd.concat([data, denoised_cluster_df, dropped_noise_df], axis = 0)
            xyz_pca = pd.concat([data, orig_noise])
            ul = set(xyz_pca['Label'].values.tolist())
            
            if -1 in ul:
                ul.remove(-1)
            if len(ul) > 0:
                ls = set(xyz_pca.loc[xyz_pca['Label'] != -1])
                if len(ls) > 0:
                    img_props, clstr_props, cluster_props_dict, xyzl = general_cluster_and_group(xyz_pca, xyz_pca['Label'].values.tolist(), fname, d2_th, d3_th)

                    if len(img_props.index) > 0:
                        img_props['Scan Parameters'] = [['Algorithm: HDBSCAN',
                                                         'min_cluster_points: ' + str(min_cluster_points),
                                                         'epsilon_threshold: ' + str(epsilon_threshold),
                                                         'min_samples: ' + str(min_samples),
                                                         'extracting_alg: ' + str(extracting_alg),
                                                         'alpha: ' + str(alpha),
                                                         'max_std_distance: ' + str(max_std_distance),
                                                         'PCA: ' + str(pca_stddev)]]
##                    for p in noise:
##                        row_num = xyz.loc[(xyz['x'] == p[0]) & (xyz['y'] == p[1]) & (xyz['z'] == p[2])].index.tolist()
##                        for n in row_num:
##                            xyz.at[n, 'Label'] = -1
                
                else:
                    img_props = pd.DataFrame()
                    clstr_props = pd.DataFrame()
                    cluster_props_dict = {}
            
            else:
                img_props = pd.DataFrame()
                clstr_props = pd.DataFrame()
                cluster_props_dict = {}

        else:
            img_props, clstr_props, cluster_props_dict, xyzl = general_cluster_and_group(xyz, labels, fname, d2_th, d3_th)

            if len(img_props.index) > 0:
                img_props['Scan Parameters'] = [['Algorithm: HDBSCAN',
                                                 'min_cluster_points: ' + str(min_cluster_points),
                                                 'epsilon_threshold: ' + str(epsilon_threshold),
                                                 'min_samples: ' + str(min_samples),
                                                 'extracting_alg: ' + str(extracting_alg),
                                                 'alpha: ' + str(alpha),
                                                 'max_std_distance: ' + str(max_std_distance),
                                                 'PCA: No PCA']]
##            for p in noise:
##                row_num = xyz.loc[(xyz['x'] == p[0]) & (xyz['y'] == p[1]) & (xyz['z'] == p[2])].index.tolist()
##                for n in row_num:
##                    xyz.at[n, 'Label'] = -1

        return img_props, clstr_props, cluster_props_dict, xyzl

    except BaseException as be:
        print(be)




def focal_cluster_and_group(xyz,
                            sigma = 55,
                            minL = 1,
                            minC = 25,
                            minPC = 0,
                            fname = '',
                            pca_stddev = None,
                            d2_th = 0.0,
                            d3_th = 0.0):
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

    labels = all_locs_df['Label'].values.tolist()
    orig_noise = all_locs_df.loc[all_locs_df['Label'] == -1]
    fin_locs_df = all_locs_df.copy()
    try:
        if pca_stddev != None:
            data = pd.DataFrame()
            for l in set(labels):
                if l != -1:
                    print('Label: ', l)
                    cluster_df = all_locs_df.loc[all_locs_df['Label'] == l]
                    cluster = cluster_df[['x', 'y', 'z']].to_numpy()
                    denoised_cluster, dropped_noise = pca_outliers_and_axes(cluster, pca_stddev)
                    denoised_cluster_df = pd.DataFrame(denoised_cluster, columns = ['x', 'y', 'z'])
                    dropped_noise_df = pd.DataFrame(dropped_noise, columns = ['x', 'y', 'z'])
                    denoised_cluster_df['Label'] = [l]*len(denoised_cluster_df.index)
                    dropped_noise_df['Label'] = [-1]*len(dropped_noise_df.index)
                    data = pd.concat([data, denoised_cluster_df, dropped_noise_df], axis = 0)

            all_locs_pca = pd.concat([data, orig_noise])
            ul = set(all_locs_pca['Label'].values.tolist())
            if -1 in ul:
                ul.remove(-1)
            if len(ul) > 0:
                ls = set(all_locs_pca.loc[all_locs_pca['Label'] != -1])
                if len(ls) > 0:
                    img_props, clstr_props, cluster_props_dict, fin_locs_df = general_cluster_and_group(all_locs_pca, all_locs_pca['Label'].values.tolist(), fname, d2_th, d3_th)

                    if len(img_props.index) > 0:
                        img_props['Scan Parameters'] = [['Algorithm: FOCAL',
                                                         'Sigma: ' + str(sigma),
                                                         'minL: ' + str(minL),
                                                         'minC: ' + str(minC),
                                                         'minPC: ' + str(minPC),
                                                         'PCA: ' + str(pca_stddev)]]

    ##                for p in noise:
    ##                    row_num = all_locs_df.loc[(all_locs_df['x'] == p[0]) & (all_locs_df['y'] == p[1]) & (all_locs_df['z'] == p[2])].index.tolist()
    ##                    for n in row_num:
    ##                        all_locs_df.at[n, 'Label'] = -1
                    
                else:
                    img_props = pd.DataFrame()
                    clstr_props = pd.DataFrame()
                    cluster_props_dict = {}
            
            else:
                img_props = pd.DataFrame()
                clstr_props = pd.DataFrame()
                cluster_props_dict = {}
        else:
            img_props, clstr_props, cluster_props_dict, fin_locs_df = general_cluster_and_group(all_locs_df, labels, fname, d2_th, d3_th)

            if len(img_props.index) > 0:
                img_props['Scan Parameters'] = [['Algorithm: FOCAL',
                                                 'Sigma: ' + str(sigma),
                                                 'minL: ' + str(minL),
                                                 'minC: ' + str(minC),
                                                 'minPC: ' + str(minPC),
                                                 'PCA: No PCA']]

    ##        for p in noise:
    ##            row_num = all_locs_df.loc[(all_locs_df['x'] == p[0]) & (all_locs_df['y'] == p[1]) & (all_locs_df['z'] == p[2])].index.tolist()
    ##            for n in row_num:
    ##                all_locs_df.at[n, 'Label'] = -1
                

        return img_props, clstr_props, cluster_props_dict, fin_locs_df

    except BaseException as be:
        print(be)
    

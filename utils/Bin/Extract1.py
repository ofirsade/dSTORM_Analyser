import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn import metrics
from sklearn.decomposition import PCA
import plotly
import math
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ast
import scipy
from scipy.spatial import distance_matrix, ConvexHull
import statistics
import functools
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from traceback import format_exc
import torch
import torch.utils.data as data
import time

def pca_outliers_and_axes(np_array, stddev_factor = 1.0):
    noise_reduce = False
    try:
        points_mat = np.matrix(np_array)
        transposed_mat = points_mat.T
        dimensions_mean = [sum(column) / len(column) for column in transposed_mat.tolist()]

        normed_transposed_mat = np.matrix(np.stack([[a - dimensions_mean[i] for a in column] for i, column in enumerate(transposed_mat.tolist())]))
        covariance_matrix = np.cov(normed_transposed_mat)

        ### eigen vectors should be orthogonal
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_values = [(e, i) for i, e in enumerate(eigen_values)]

        ### sort eigen values descending, and create a feature vector accordingly
        eigen_values.sort(reverse = True, key = lambda a:a[0])

        ### returns a transposed matrix of eigen vectors!
        feature_vec = np.stack([eigen_vectors[:,i] for e, i in eigen_values])

        transformed_data = feature_vec * normed_transposed_mat

        ### sanity check
        if (int((transformed_data[0]).mean()) != 0):
            raise BaseException("Some error has occurred during the PCA process")

        ### actual noise reduction, look at PC1 which has the highest explained variance,
        ### filter out all points which are larger than stddev_factor * std
        pc_one = transformed_data.tolist()[0]
        std = transformed_data[0].std()
        noise = [idx for idx,val in enumerate(pc_one) if np.abs(val) >= std * stddev_factor]
        noise_data = []

        if (len(noise) > 0):
            print("Noise reduction: %d points dropped due to being %f times higher than std (second PC)" % (len(noise), stddev_factor))
            noise_reduce = True

        ### drop noise
        noise_data = np.take(transformed_data, noise, axis = 1)
        transformed_data = np.delete(transformed_data, noise, axis = 1)

        if (len(transformed_data) > 2):
            pc_two = transformed_data.tolist()[1]
            std = transformed_data[1].std()
            noise = [idx  for idx,val in enumerate(pc_two) if np.abs(val) >= std * stddev_factor]

            if (len(noise) > 0):
                print("Noise reduction: %d points dropped due to being %f times higher than std" % (len(noise), stddev_factor))
                noise_reduce = True
            ### drop additional noise
            noise_data.append(transformed_data[noise])
            transformed_data = np.delete(transformed_data, noise, 1)

        #restore mean in reduced original data
        original_data_reduced = np.matrix(feature_vec).I * transformed_data
        restored = np.stack([[a + dimensions_mean[i] for a in column] for i, column in enumerate(original_data_reduced.tolist())])
        
        return (restored.T, noise_reduce)#, noise_data
    
    except Exception as e:
        print("Error ocurred during PCA noise reduction!")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)
        print(e)


def PolygonArea(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def PolygonSort(corners):
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key = lambda tup: tup[2])
    return [(x,y) for (x,y,an) in cornersWithAngles]


def div_lst_to_clusters(no_noise_lst, unique_labels):
    """
    Divides list of labeled points into a dictionary.
    The dictionary's keys represent cluster labels.
    The dictionary's values represent points belonging to the cluster with relevant label = key.
    """
    clstrs_dict = {}
    for i in unique_labels:
        label = str(i)
        clstrs_dict[label] = []
    for p in no_noise_lst:
        clstrs_dict[str(int(p[3]))].append(p)
    return clstrs_dict

def calc_localizations(clusters):
    """
    Calculates the mean and median number of localizations per cluster.
    """
    pnts_lst = []
    for cluster in clusters.values():
        pnts_lst.append(len(cluster))
    median_loc = statistics.median(pnts_lst)
    mean_loc = statistics.mean(pnts_lst)
    return mean_loc, median_loc

'''
def calc_3D_poly_vol_density(clusters, full_lst, no_noise_lst, density_th3):
    """
    Calculates the mean and median densities of the 3D convex hull containing each cluster.
    """
    densities_lst = []
    vol_lst = []
    vol_dens_dict = {}
    labels = list(set(clusters.keys()))
    for cluster_lst in clusters.values():
        points = np.array([point[:3] for point in cluster_lst])
        hull = scipy.spatial.ConvexHull(points)
        vol = hull.volume
        n = len(cluster_lst)
        temp_density = 1000 * n / vol
        densities_lst.append(temp_density)
        vol_lst.append(temp_area)
        vol_dens_dict[str(cluster[0][3])] = [vol, temp_density]

    clstrs_dict = clusters.copy()
    if density_th3 > 0.0:
        for cluster in clusters.values():
            key = str(cluster[0][3])
            dens = vol_dens_dict[key][1]
            if dens < density_th3:
                for p in cluster:
                    if p[3] == key:
                        full_lst.remove(p)
                        p[3] = -1
                        full_lst.append(p)
                        no_noise_lst.remove(p)
                del clstrs_dict[key]
                print('\nCluster with label ', key, ' dropped because its 3D density = ', dens, ' < ', density_th3)
                labels.remove(ast.literal_eval(key))

    mean_vol = statistics.mean(vol_lst)
    median_vol = statistics.median(vol_lst)
    mean_dens = statistics.mean(densities_lst)
    median_dens = statistics.median(densities_lst)

    return mean_vol, median_vol, mean_dens, median_dens, labels


def calc_2D_poly_size_density(clusters, full_lst, no_noise_lst, density_th2):
    """
    Calculates the mean & median sizes & densities of the xy convex hull containing each cluster.
    """
    densities_lst = []
    sizes_lst = []
    size_dens_dict = {}
    labels = list(set(clusters.keys()))
    for cluster in clusters.values():
        points = np.array([item[:2] for item in cluster])
        hull = ConvexHull(points)
        corners = list(set(functools.reduce(lambda x,y: x+y,
                                            [[(a,b) for a,b in x] for x in points[hull.simplices]])))
        temp_area = PolygonArea(PolygonSort(corners))
        n = len(cluster)
        temp_density = n / temp_area
        densities_lst.append(temp_density)
        sizes_lst.append(temp_area)
        size_dens_dict[str(cluster[0][3])] = [temp_area, temp_density]

    clstrs_dict = clusters.copy()
    if density_th2 > 0.0:
        for cluster in clusters.values():
            key = str(cluster[0][3])
            dens = size_dens_dict[key][1]
            if dens < density_th2:
                for p in cluster:
                    if p[3] == key:
                        full_lst.remove(p)
                        p[3] = -1
                        full_lst.append(p)
                        no_noise_lst.remove(p)
                del clstrs_dict[key]
                print('\nCluster with label ', key, ' dropped because its 2D density = ', dens, ' < ', density_th2)
                labels.remove(ast.literal_eval(key))
    cluster_num = len(labels)
    clusters = clstrs_dict
    

    mean_size = statistics.mean(sizes_lst)
    median_size = statistics.median(sizes_lst)
    mean_dens = statistics.mean(densities_lst)
    median_dens = statistics.median(densities_lst)

    return mean_size, median_size, mean_dens, median_dens, labels


def calc_polygon_radius(clusters):
    """
    Approximates the radius of the xy (2D) covex hull containing each cluster.
    """
    radii = []
    for cluster in clusters.values():
        points = np.array([item[:2] for item in cluster])
        hull = ConvexHull(points)
        perimeter = hull.area
        size = hull.volume
        radius = 2 * size / perimeter
        radii.append(radius)
    
    mean_radius = statistics.mean(radii)
    median_radius = statistics.median(radii)

    return mean_radius, median_radius
'''

def calc_3D_poly_vol_density(cluster, label, thd_hull):
    """
    Calculates the mean and median densities of the 3D convex hull containing each cluster.
    """
    n = len(cluster)
    vol = thd_hull.volume
    density = 1000 * n / vol

    return [vol, density]


def calc_2D_poly_area_density(cluster, label, td_hull):
    """
    Calculates the mean & median sizes & densities of the xy convex hull containing each cluster.
    """
    points = np.array([item[:2] for item in cluster])
    corners = list(set(functools.reduce(lambda x,y: x+y,
                                        [[(a,b) for a,b in x] for x in points[td_hull.simplices]])))
    area = PolygonArea(PolygonSort(corners))
    n = len(cluster)
    density = float(n / area)

    return [area, density]


def calc_polygon_radius(td_hull):
    """
    Approximates the radius of the xy (2D) covex hull containing each cluster.
    """
    perimeter = td_hull.area
    area = td_hull.volume
    radius = float(2 * area / perimeter)

    return radius

def get_unassigned(labeled_pts_lst):
    '''
    @param labeles_pts_lst: list of lists (nx4), all localisations with cluster labels
    ***********************************************************************
    @ret unassigned: df, all localisations not assigned to a cluster
    '''
    noise = []
    for p in labeled_pts_lst:
        if p[3] == -1:
            noise.append(p)
    unassigned = pd.DataFrame(noise.T, columns = ['x', 'y', 'z', 'label'])
    return unassigned


def extract_AP(labeled_locs, pca_stddev, td_density_th = -1.0, thd_density_th = -1.0):
    """
    @param labeled_pts: df, x,y,z coordinates and cluster label for all localisations in a single dataset
    @param pca_stddev: float, a user defined stadard deviation for PCA.
                       if the user didn't define it, it is set to 1.0
    @param td_density_th: float, a user defined 2D density threshold
    @param thd_density_th: float, a user defined 3D density threshold
    ----------
    @ret clstrs_df_rows: df, contains a list of attributes for each cluster -
                        localisations in cluster, label, number of points, 2D convex hull, 2D polygon area,
                        2D polygon density, 2D polygon radius, 2D polygon perimeter, 3D polygon volume,
                        3D polygon density, pca components, pca mean, pca std, pca size
    """
    res_lst = []
    number_of_locs = len(labeled_locs.index)
    clustered_locs = labeled_pts.loc[labeled_locs['Label']!= -1]
    number_of_locs_assigned_to_clusters = len(clustered_locs.index)
    lst_of_labels = labeled_locs['Label'].values.tolist()
    s = set(lst_of_labels)
    labeled_locs_lst = labeled_locs.to_numpy()
    no_noise_lst = clustered_locs.values.tolist()
    clusters = div_lst_to_clusters(no_noise_lst, int(max(s))+1) # Dict: key = label, value = localisation (x,y,z,label)
    
    unassigned = get_unassigned(labeled_locs_lst)
    clstrs_dict = clusters.copy()
    clstrs_df = pd.DataFrame()
    
    for label,cluster in clstrs_dict.items():
        clstr_df_row = {}
        clstr_df_row['label'] = label
        clstr_df_row['cluster'] = cluster
        points = np.array([item[:2] for item in cluster])
        nr, noise_reduce = pca_outliers_and_axes(points, pca_stddev)
        if (noise_reduce):
            clstr_df_row['noise reduced clusters'] = nr
            pca_pc = nr
            xy_plane_pc = nr[:,[0,1]]
        else:
            clstr_df_row['noise reduced clusters'] = []
            xy_plane_pc = cluster[:,[0,1]]
        
        td_hull = ConvexHull(xy_plane_pc)
        thd_convex_hull = ConvexHull(pca_pc)
        clstr_df_row['number of points'] = pca_pc.shape[0]
        clstr_df_row['2D convex hull'] = xy_plane_pc[td_hull.simplices]
        clstr_df_row['2D polygon area'], clstr_df_row['2D polygon density'] = calc_2D_poly_area_density(xy_plane_pc, label, td_hull)
        clstr_df_row['2D polygon radius'], clstr_df_row['2D polygon perimeter'] = calc_polygon_radius(td_hull)
        clstr_df_row['3D volume'], clstr_df_row['3D density'] = calc_3D_poly_vol_density(pca_pc, label, thd_hull)

        if (td_density_th > -1.0):
            if (clstr_df_row['2D polygon density'] < td_density_th):
                print('Dropping cluster due to 2D density (%f < %f)' % (clstr_df_row['2D polygon density'], td_density_th))
                for p in cluster:
                    p[3] = -1
                unassigned = pd.concat([unassigned, cluster])
                del clstrs_dict[label]
                continue
        
        if (thd_density_th > -1.0):
            if (clstr_df_row['3D density'] < thd_density_th):
                print('Dropping cluster due to 3D density (%f < %f)' % (clstr_df_row['3D density'], thd_density_th))
                for p in cluster:
                    p[3] = -1
                unassigned = pd.concat([unassigned, cluster])
                del clstrs_dict[label]
                continue

        pca = PCA()
        pca.fit(pca_pc)
        clstr_df_row['pca_components'] = pca.components_
        # per feature empirical mean; equal to X.mean(axis = 0)
        clstr_df_row['pca_mean'] = pca.mean_
        # square root of the amount of variance explained by each of the selected components
        clstr_df_row['pca_std'] = np.sqrt(pca.explained_variance_)
        # figure out what pca size is...?
        clstr_df_row['pca_size'] = np.sqrt(np.prod(clstr_df_row['pca_std']))
        #accumulative_size += clstr_df_row['2D polygon size']

        #clstrs_df_rows.append(clstr_df_row)
        clstrs_df = pd.concat([clstrs_df, clstr_df_row])

    return clstrs_df, number_of_localizations, number_of_locs_assigned_to_clusters
    
    '''
        temp_lst = [filename, s - 1, number_of_localizations,
                    number_of_localizations_assigned_to_clusters,
                    mean_loc, median_loc,
                    flat_poly_mean_size, flat_poly_median_size,
                    mean_density, median_density,
                    mean_reduced_polygon_density,
                    median_reduced_polygon_density,
                    mean_reduced_polygon_size,
                    median_reduced_polygon_size]
        
        res_lst.append(temp_lst)
    
    results_df = pd.DataFrame(res_lst,
                                  columns=['Filename', 'Number of Clusters',
                                           'Number of Localizations',
                                           'Number of Localizations Assigned to Clusters',
                                           'Cluster Mean Localizations',
                                           'Cluster Median Localizations',
                                           'Flat Polygon Mean Size',
                                           'Flat Polygon Median Size',
                                           'Mean Polygon Density',
                                           'Median Polygon Density',
                                           'Mean Reduced Polygon Density',
                                           'Median Reduced Polygon Density',
                                           'Mean Reduced Polygon Size',
                                           'Median Reduced Polygon Size'])
    '''






    

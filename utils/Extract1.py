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



def get_outliers_and_axis_reduction_pca(np_array, stddev_factor=1.5):
    try:
        points_mat = np.matrix(np_array)
        transposed_mat = points_mat.T
        dimensions_mean = [sum(column) / len(column) for column in transposed_mat.tolist()]

        normed_transposed_mat = np.matrix(np.stack([[a - dimensions_mean[i] for a in column] for i, column in enumerate(transposed_mat.tolist())]))
        #normed_mat = normed_transposed_mat.T
        covariance_matrix = np.cov(normed_transposed_mat)

        # eigen vectors should be orthogonal
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_values = [(e, i) for i, e in enumerate(eigen_values)]

        # sort eigen values descending, and create a feature vector according
        eigen_values.sort(reverse=True, key=lambda a: a[0])

        # alreay returns a transposed matrix of eigen vectors!!!!
        feature_vec = np.stack([eigen_vectors[:,i] for e, i in eigen_values])

        transformed_data = feature_vec * normed_transposed_mat
        reduced_data = None

        if (len(feature_vec) > 2):
            reduced_data = feature_vec[:2] * normed_transposed_mat
            reduced_data = reduced_data.T

        # just some sanity
        if (int((transformed_data[0]).mean()) != 0):
            raise BaseException("Some error has occurred during computation") 

        # actual noise reduction, look at PC1 which has the highest explained variance,
        # and filter out all points which are bigger than stddev_factor * std
        pc_one = transformed_data.tolist()[0]
        std = transformed_data[0].std()
        noise = [idx  for idx,val in enumerate(pc_one) if np.abs(val) >= std * stddev_factor]

        if (len(noise) > 0):
            print("Noise reduction: %d points dropped due to being %f times higher than std (second PC)" % (len(noise), stddev_factor))

        # drop noise
        transformed_data = np.delete(transformed_data, noise, 1)

        if (len(transformed_data) > 2): 
            pc_two = transformed_data.tolist()[1]
            std = transformed_data[1].std()
            noise = [idx  for idx,val in enumerate(pc_two) if np.abs(val) >= std * stddev_factor]

            if (len(noise) > 0):
                print("Noise reduction: %d points dropped due to being %f times higher than std" % (len(noise), stddev_factor))

            # drop noise
            transformed_data = np.delete(transformed_data, noise, 1)

        #restore mean in reduced original data
        original_data_reduced = np.matrix(feature_vec).I * transformed_data
        restored = np.stack([[a + dimensions_mean[i] for a in column] for i, column in enumerate(original_data_reduced.tolist())])
        return (restored.T, reduced_data)
    except Exception as e:
        print("Error ocurred during noise reduction!")
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

def get_axis_reduction_pca(np_array):
        """
        INSERT DESCRIPTION
        """
        points_mat = np.matrix(np_array)
        transposed_mat = points_mat.T
        dimensions_mean = [sum(column) / len(column) for column in transposed_mat.tolist()]

        normed_transposed_mat = np.matrix(np.stack([[a - dimensions_mean[i] for a in column]
                                                    for i, column in enumerate(transposed_mat.tolist())]))
        covariance_matrix = np.cov(normed_transposed_mat)

        # eigen vectors should be orthogonal
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_values = [(e,i) for i, e in enumerate(eigen_values)]

        # sort eigen values in descending order and create a feature vector accordingly
        eigen_values.sort(reverse = True, key = lambda a : a[0])

        # alread returns a transposed matrix of eigen vectors
        feature_vec = np.stack([eigen_vectors[:,i] for e, i in eigen_values])

        reduced_data = None

        if (len(feature_vec) > 2):
            reduced_data = feature_vec[:2] * normed_transposed_mat
            reduced_data = reduced_data.T

        return reduced_data

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



def extract_AP(labeled_pts):
    """
    @param labeled_pts: df, x,y,z coordinates and cluster label for all localisations in a single dataset
    ----------
    @ret *:
    """
    res_lst = []
    number_of_localizations = len(labeled_pts.index)
    non_noisy_pts = labeled_pts.loc[labeled_pts['Label']!= -1]
    number_of_locs_assigned_to_clusters = len(non_noisy_pts.index)
    lst_of_labels = labeled_pts['Label'].values.tolist()
    s = set(lst_of_labels)
    labeled_pts_lst = labeled_pts.to_numpy()
    no_noise_lst = non_noisy_pts.values.tolist()
    clusters = div_lst_to_clusters(no_noise_lst, int(max(s))+1)
    mean_size, median_size, mean_density2, median_density2, labels2 = calc_2D_poly_size_density(clusters, labeled_pts_lst, no_noise_lst, density_th2)
    mean_vol, median_vol, mean_density3, median_density3, labels3 = calc_3D_poly_size_density(clusters, labeled_pts_lst, no_noise_lst, density_th3)
    mean_loc, median_loc = calc_localizations(clusters)
    calc_polygon_radius(clusters)
    """
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
    """
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
    







    

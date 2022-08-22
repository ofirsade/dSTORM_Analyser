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


def calc_3D_polygon_density(clusters):
    """
    Calculates the mean and median densities of the 3D convex hull containing each cluster.
    """
    densities_lst = []
    for cluster_lst in clusters.values():
        print("CLUSTER: ", cluster_lst[0][3])
        points = np.array([point[:3] for point in cluster_lst])
        hull = scipy.spatial.ConvexHull(points)
        #vol = hull.volume
        area = hull.area
        n = len(cluster_lst)
        #temp_density = (1000 * n) / vol
        temp_density = (1000 * n) / area
        densities_lst.append(temp_density)
    mean_dens = statistics.mean(densities_lst)
    median_dens = statistics.median(densities_lst)

    return mean_dens, median_dens

def calc_polygon_density(self):
        """
        Calculates the mean and median densities of the xy convex hull containing each cluster.
        """
        densities_lst = []
        for cluster in self.clusters.values():
            points = np.array([item[:2] for item in cluster])
            hull = ConvexHull(points)
            corners = list(set(functools.reduce(lambda x,y: x+y,
                                                [[(a,b) for a,b in x] for x in points[hull.simplices]])))
            temp_area = PolygonArea(PolygonSort(corners))
            n = len(cluster)
            temp_density = (1000 * n) / temp_area
            densities_lst.append(temp_density)
        mean_dens = statistics.mean(densities_lst)
        median_dens = statistics.median(densities_lst)
        return mean_dens, median_dens

def calc_red_poly_density_size(clusters):
    """
    Calculates the mean and median sizes and densities of reduced convex hull containing each cluster.
    """
    reduced_poly_size = []
    reduced_poly_density = []
    for cluster in clusters.values():
        points = np.array([item[:3] for item in cluster])
        reduced_cluster = get_axis_reduction_pca(points)
        if reduced_cluster is not None:
            reduced_convex_hull = ConvexHull(reduced_cluster)
            corners = list(set(functools.reduce(lambda x,y: x + y,
                                                [[(a.tolist()[0][0], a.tolist()[0][1]) for a in x]
                                                 for x in reduced_cluster[reduced_convex_hull.simplices]])))
            temp_poly_size = PolygonArea(PolygonSort(corners))
            reduced_poly_size.append(temp_poly_size)
            temp_density = float((1000 * len(reduced_cluster)) / temp_poly_size)
            reduced_poly_density.append(temp_density)
    mean_size = statistics.mean(reduced_poly_size)
    mean_dens = statistics.mean(reduced_poly_density)
    median_size = statistics.median(reduced_poly_size)
    median_dens = statistics.median(reduced_poly_density)
    
    return mean_dens, mean_size, median_dens, median_size

def calc_poly_surrounding_flat_cluster_size(clusters):
    """
    Calculates the mean and median size of the surrounding flat polygon of each cluster.
    """
    sizes_lst = []
    for cluster in clusters.values():
        points = np.array([item[:2] for item in cluster])
        hull = ConvexHull(points)
        temp_area = hull.area
        sizes_lst.append(temp_area)
    mean_size = statistics.mean(sizes_lst)
    median_size = statistics.median(sizes_lst)

    return mean_size, median_size

def calc_poly_surrounding_flat_cluster_size1(self):
        """
        Calculates the mean and median size of the surrounding flat polygon of each cluster.
        """
        sizes_lst = []
        for cluster in self.clusters.values():
            points = np.array([item[:2] for item in cluster])
            hull = ConvexHull(points)
            corners = list(set(functools.reduce(lambda x,y: x+y,
                                                [[(a,b) for a,b in x] for x in points[hull.simplices]])))
            temp_area = PolygonArea(PolygonSort(corners))
            sizes_lst.append(temp_area)
        mean_size = statistics.mean(sizes_lst)
        median_size = statistics.median(sizes_lst)
        return mean_size, median_size


def extract_AP(res):
    """
    @param res:
    ----------
    @ret *:
    """
    res_lst = []
    for lst in res:
        filename = lst[0]
        print(str(filename))
        
        pts_df = pd.read_csv(input_path + ".csv", usecols = cols_list)
        number_of_localizations = len(pts_df.index)
        non_noisy_df = pts_df.loc[pts_df['Label']!= -1]
        number_of_localizations_assigned_to_clusters = len(non_noisy_df.index)
        lst_of_labels = pts_df['Label'].values.tolist()
        s = set(lst_of_labels)
        lst = non_noisy_df.values.tolist()
        clusters = div_lst_to_clusters(lst, int(max(s))+1)
        mean_loc, median_loc = calc_localizations(clusters)
        mean_density, median_density = calc_polygon_density(clusters)
        mean_reduced_polygon_density,mean_reduced_polygon_size, median_reduced_polygon_density,median_reduced_polygon_size = calc_red_poly_density_size(clusters)
        flat_poly_mean_size, flat_poly_median_size = calc_poly_surrounding_flat_cluster_size1(clusters)
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
    







    

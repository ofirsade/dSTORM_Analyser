import pandas as pd
import numpy as np
from collections import Counter
from sklearn import metrics
import math
import scipy
import json
import scipy
from scipy.spatial import distance_matrix, ConvexHull
import os
import statistics
import functools
import ast
##from sklearn.decomposition import PCA
import sys
from scipy.spatial.distance import cdist


def pca_outliers_and_axes(np_array, stddev_factor): # This method was originally written by Itay Talpir 
    try:
        points_mat = np.matrix(np_array)
        transposed_mat = points_mat.T
        dimensions_mean = [sum(column) / len(column) for column in transposed_mat.tolist()]
        noise_two = None

        normed_transposed_mat = np.matrix(np.stack([[a - dimensions_mean[i] for a in column] for i, column in enumerate(transposed_mat.tolist())]))
        covariance_matrix = np.cov(normed_transposed_mat)

        ### eigen vectors should be orthogonal
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_values = [(e, i) for i, e in enumerate(eigen_values)]

        ### sort eigen values descending, and create a feature vector accordingly
        eigen_values.sort(reverse = True, key = lambda a:a[0])

        ### returns a transposed matrix of eigen vectors
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

        if (len(noise) > 0):
            print("Noise reduction: %d points dropped due to being %f times higher than std" % (len(noise), stddev_factor))

        ### drop noise
##        noise_data = np.take(transformed_data, noise, axis = 1)
        transformed_data = np.delete(transformed_data, noise, axis = 1)

        if (len(transformed_data) > 2):
            pc_two = transformed_data.tolist()[1]
            std = transformed_data[1].std()
            noise_two = [idx  for idx,val in enumerate(pc_two) if np.abs(val) >= std * stddev_factor]

            if (len(noise_two) > 0):
                print("Noise reduction: %d points dropped due to being %f times higher than std (second PC)" % (len(noise_two), stddev_factor))

            ### drop additional noise
##            noise_data.append(transformed_data[noise])
            transformed_data = np.delete(transformed_data, noise_two, 1)
        if noise_two != None:
            noise = noise + noise_two
        #restore mean in reduced original data
        original_data_reduced = np.matrix(feature_vec).I * transformed_data
        restored = np.stack([[a + dimensions_mean[i] for a in column] for i, column in enumerate(original_data_reduced.tolist())])

        restored_noise = [item for item in np_array if item not in restored.T]
        
        return restored.T, restored_noise
    
    except Exception as e:
        print("Error ocurred during PCA noise reduction!")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)
        print(e)
        ret1 = None
        ret2 = None
        return ret1, ret2


# ************************************************************** Polygon ************************************************************** #

def PolygonArea(corners):
    """
    Calculates Polygon surrounding cluster area.
    Param:
     ** corners - convex hull's simplices (type numpy array).
    """
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def PolygonSort(corners):
    """
    Sorts polygon's (surrounding cluster) corners which are represented by simplices.
    Param:
     ** corners - convex hull's simplices (type numpy array).
    """
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key = lambda tup: tup[2])
    return [(x,y) for (x,y,an) in cornersWithAngles]


def calc_volumes(clusters):
    """
    Calculates the volume of the convex hull containing each cluster.
    Param:
     ** no_noise_lst: labeled localizations which were assigned to clusters (type list).
    """
    vols_dict = {}
    for label,cluster in clusters.items():
        clstr = []
        for p in cluster:
            p1 = p[0:3]
            clstr.append(p1)
        points = np.array(clstr)
        volume = ConvexHull(points).volume
        vols_dict[label] = volume
    return vols_dict

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

def calc_size_density(clusters):
    """
    Calculates the mean and median densities of the xy convex hull containing each cluster.
    """
    densities_lst = []
    sizes_lst = []

    for cluster in clusters.values():
        points = np.array([item[:2] for item in cluster])
        hull = ConvexHull(points)
        corners = list(set(functools.reduce(lambda x,y: x+y,
                                            [[(a,b) for a,b in x] for x in points[hull.simplices]])))
        temp_area = PolygonArea(PolygonSort(corners))
        n = len(cluster)
        temp_density = n / temp_area
        densities_lst.append(temp_density)

    mean_dens = statistics.mean(densities_lst)
    median_dens = statistics.median(densities_lst)

    return mean_dens, median_dens, densities_lst


def calc_3D_polygon_density(clusters):
    """
    Calculates the mean and median densities of the 3D convex hull containing each cluster.
    """
    densities_lst = []
    for cluster_lst in clusters.values():
        points = np.array([point[:3] for point in cluster_lst])
        hull = scipy.spatial.ConvexHull(points)
        vol = hull.volume
        n = len(cluster_lst)
        temp_density = (1000 * n) / vol
        densities_lst.append(temp_density)
    mean_dens = statistics.mean(densities_lst)
    median_dens = statistics.median(densities_lst)

    return mean_dens, median_dens, densities_lst


def calc_polygon_radius(clusters):
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

    return mean_radius, median_radius, radii

def calc_cluster_hw(cluster):

    points_3d = np.array([point[:3] for point in cluster])
    hull = scipy.spatial.ConvexHull(points_3d)
    corners = list(set(functools.reduce(lambda x,y: x+y,
                                        [[(a,b,c) for a,b,c in x] for x in points_3d[hull.simplices]])))
    hdist = cdist(corners, corners, metric='euclidean')
    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    bpair = [corners[bestpair[0]],corners[bestpair[1]]]
    print(bpair)

    p1 = bpair[0]
    p2 = bpair[1]

    x_dist = abs(p1[0]-p2[0])
    y_dist = abs(p1[1]-p2[1])

    if x_dist > y_dist:
        major_axis = 0
        length = x_dist
    else:
        major_axis = 1
        length = y_dist

    lengths.append(length)
    inc_size = length / 3

    inc0 = min(p1[major_axis], p2[major_axis])
    inc1 = inc0 + inc_size
    inc2 = inc1 + inc_size
    inc3 = max(p1[major_axis], p2[major_axis])
    incs = [inc0, inc1, inc2, inc3]
    minor_axis = 1-major_axis

    inc0_points = np.array([p for p in corners if p[major_axis] <= inc1])
    inc1_points = np.array([p for p in corners if all((p[major_axis] > inc1) & (corners[major_axis] <= inc2))])
    inc2_points = np.array([p for p in corners if all((p[major_axis] > inc2) & (corners[major_axis] <= inc3))])

    width0 = -1
    width1 = -1
    width2 = -1
    if len(inc0_points) > 0:
        inc0_max_idx = np.argmax(inc0_points, axis = 0)
        inc0_min_idx = np.argmin(inc0_points, axis = 0)

        inc0_max_minora = (inc0_points[inc0_max_idx])[minor_axis]
        inc0_min_minora = (inc0_points[inc0_min_idx])[minor_axis]

        width0 = abs(inc0_max_minora[minor_axis] - inc0_min_minora[minor_axis])
    if len(inc1_points) > 0:
        inc1_max_idx = np.argmax(inc1_points, axis = 0)
        inc1_min_idx = np.argmin(inc1_points, axis = 0)

        inc1_max_minora = (inc1_points[inc1_max_idx])[minor_axis]
        inc1_min_minora = (inc1_points[inc1_min_idx])[minor_axis]

        width1 = abs(inc1_max_minora[minor_axis] - inc1_min_minora[minor_axis])
    if len(inc2_points) > 0:
        inc2_max_idx = np.argmax(inc2_points, axis = 0)
        inc2_min_idx = np.argmin(inc2_points, axis = 0)

        inc2_max_minora = (inc2_points[inc2_max_idx])[minor_axis]
        inc2_min_minora = (inc2_points[inc2_min_idx])[minor_axis]

        width2 = abs(inc2_max_minora[minor_axis] - inc2_min_minora[minor_axis])
    widths = [width0, width1, width2]
    widths.remove(-1)

    width = statistics.mean(widths)

    print('\nCluster Widths:\n', width0, '\n', width1, '\n', width2)

    return length, width
        
        
        

def calc_all(clusters, d2_th, d3_th):
    """
    Calculates the volume of the convex hull containing each cluster.
    @Params:
    @@ no_noise_lst: labeled localizations which were assigned to clusters (type list).
    @@ d2_th - float: 2D density threshold
    @@ d3_th - float: 3D density threshold
    """
    gen_lst = []
    vols_list = []
    loc_list = []
    dens_2d_list = []
    dens_3d_list = []
    radii_list = []
    lengths_list = []
    widths_list = []
##    zs_list = []
    mean_vol = None
    med_vol = None
    mean_locs = None
    med_locs = None
    mean_dens_2d = None
    med_dens_2d = None
    mean_radius = None
    med_radius = None
    mean_dens_3d = None
    med_dens_3d = None
    cluster_props_dict = dict()
    noise = []
    if len(clusters.values()) > 0:
        for label,cluster in clusters.items():

            points_3d = np.array([point[:3] for point in cluster])

            # Calculate the volume of current cluster
            volume = ConvexHull(points_3d).volume

            # Count the number of localisations in current clusster
            loc_num = len(points_3d)

            # Calculate the 2D density of current cluster
            points_2d = np.array([item[:2] for item in cluster])
            hull = ConvexHull(points_2d)
            corners = list(set(functools.reduce(lambda x,y: x+y,
                                                [[(a,b) for a,b in x] for x in points_2d[hull.simplices]])))
            temp_area = PolygonArea(PolygonSort(corners))
            density_2d = (1000 * loc_num) / temp_area
            if density_2d >= d2_th:
                dens_2d_list.append(density_2d)
            else:
                noise.append(label)
                print('Cluster ', label, ' was dropped due to 2D density < ', d2_th)
                continue

            # Calculate 2D radius
##            hull = ConvexHull(points_2d)
            perimeter = hull.area
            size = hull.volume
            radius = 2 * size / perimeter
            
            # Calculate 3D density
            hull = scipy.spatial.ConvexHull(points_3d)
            density_3d = (1000 * loc_num) / volume
            if density_3d >= d3_th:
                dens_3d_list.append(density_3d)
            else:
##                for p in points_3d:
##                    p = np.append(p, -1)
##                    noise.append(p)
                noise.append(label)
                print('Cluster ', label, ' was dropped due to 3D density < ', d3_th)
                continue

            vols_list.append(volume)
            loc_list.append(loc_num)
            radii_list.append(radius)

##            # Calculate maximal distances in x,y,z axes
##            corners = list(set(functools.reduce(lambda x,y: x+y,
##                                                [[(a,b,c) for a,b,c in x] for x in points_3d[hull.simplices]])))
##            mx_x_dist = 0
##            mx_y_dist = 0
##            mx_z_dist = 0
##            l = len(corners)
##
##            for p1 in corners:
##                for p2 in corners:
##                    if p1 != p2:
##                        x_dist = abs(p2[0]-p1[0])
##                        if x_dist >= mx_x_dist:
##                            mx_x_dist = x_dist
##                        y_dist = abs(p2[1]-p1[1])
##                        if y_dist >= mx_y_dist:
##                            mx_y_dist = y_dist
##                        z_dist = abs(p2[2]-p1[2])
##                        if z_dist >= mx_z_dist:
##                            mx_z_dist = z_dist
##            xs_list.append(mx_x_dist)
##            ys_list.append(mx_y_dist)
##            zs_list.append(mx_z_dist)

            # Calculate maximal distances in x,y axes
            length, width = calc_cluster_hw(cluster)
            lengths_list.append(length)
            width_list.append(width)
            
            clst_lst = [int(label), loc_num, volume, radius, density_2d, density_3d, length, width, cluster]
            gen_lst.append(clst_lst)
            cluster_props_dict[label] = clst_lst[1:]

        if len(vols_list) != 0:   
            mean_vol = statistics.mean(vols_list)
            med_vol = statistics.median(vols_list)
            mean_locs = statistics.mean(loc_list)
            med_locs = statistics.median(loc_list)
            mean_dens_2d = statistics.mean(dens_2d_list)
            med_dens_2d = statistics.median(dens_2d_list)
            mean_radius = statistics.mean(radii_list)
            med_radius = statistics.median(radii_list)
            mean_dens_3d = statistics.mean(dens_3d_list)
            med_dens_3d = statistics.median(dens_3d_list)
        else:
            print('All clusters were dropped due to low densities')
            
    else:
        print('No clusters were found!')

    return gen_lst, mean_vol, med_vol, mean_locs, med_locs, mean_dens_2d, med_dens_2d, mean_radius, med_radius, mean_dens_3d, med_dens_3d, cluster_props_dict, noise


# ************************************************************************************************************************************* #
def calc_ed(pnt1, pnt2):
    """ This function calculates the Euclidean distance between 2 points
    Args:
        pnt1, pnt2 - two points with coordinates x,y
    """
    x_d = (pnt1[0] - pnt2[0])
    x_s = x_d ** 2
    y_d = (pnt1[1] - pnt2[1])
    y_s = y_d ** 2
    # 3D Version
    z_d = (pnt1[2] - pnt2[2])
    z_s = z_d ** 2
    dist = math.sqrt(x_s + y_s + z_s)
    return dist

def calc_centroid(pnt_lst, n):
    """ This function averages all points and returns the centroid of the bunch
    Args:
        pnt_lst - a list of points in a suspected cluster
        n - number of points in pnts
    """
    x_sum = 0
    y_sum = 0
    z_sum = 0
    for point in pnt_lst:
       x_sum += point[0]
       y_sum += point[1]
       z_sum += point[2]

    avg_x = x_sum / n
    avg_y = y_sum / n
    avg_z = z_sum / n
    centroid = [avg_x, avg_y, avg_z]
    return centroid

def calc_max_dist(pnt_lst, centroid):
    """
    This function finds the maximal distance from the centroid of the cluster to an existing point in pnts
    Args:
        pnts - a list of points, a suspected "cluster"
        centroid - the central point of the "cluster"
    """
    max_dist = 0
    for point in pnt_lst:
        dist = calc_ed(centroid, point)
        if dist > max_dist:
            max_dist = dist
    return max_dist

# ************************************************************** 3D CASE ************************************************************** #

def measure_vol(pnt_lst):
   vol = -1
   if len(pnt_lst):
      points = np.array(pnt_lst)
      hull = scipy.spatial.ConvexHull(points)
      vol = hull.volume
   return vol

def t_from_centroid(pnt_lst, r, centroid):
    """ This function filters the points list to points within distance t from the centroid
    Args:
        pnt_lst - a list of points
        t - the radius of the circle that contains all filtered points
        centroid - the central point of the "cluster"
    """
    filtered = []
    for point in pnt_lst:
        dist = calc_ed(centroid, point)
        if dist <= r:
            filtered.append(point)
    return filtered

def calc_density(points):
   pts_df = points[["x", "y", "z"]]
   pts_lst = pts_df.values.tolist()
   vol = measure_vol(pts_lst)
   density = -1
   if vol != -1:
      n = len(pts_lst)
      density = 100000 * n / vol
   return density

def remove_outliers(pnt_lst, centroid, radius):
   core_points_lst = []
   for point in pnt_lst:
      dist = calc_ed(point, centroid)
      if dist <= radius:
         core_points_lst.append(point)
   core_points_df = pd.DataFrame(core_points_lst, columns = ['x', 'y', 'z'])
   return core_points_df
      


################################################## DBSCAN Implementation ##################################################

def extract_AP(clusters, d2_th, d3_th):
    """
    @Param:
    @@ clusters - dict, keys are cluster labels, values are localisations assigned to cluster with label==key
    @@ d2_th - float, 2D density threshold
    @@ d3_th - float, 3D density threshold
    @Return:
    @@ prop_lst - list, cluster properties of each cluster in the dataset.
    @@ gen_lst - list, mean and median cluster properties of the image.
    @@ noise - list, points to add to non-clustered list of localisations.
    """
##    vols_dict = calc_volumes(clusters)
##    mean_loc_num, med_loc_num = calc_localizations(clusters)
##    mean_radius, med_radius, radii_lst = calc_polygon_radius(clusters)
##    mean_2d_dens, med_2d_dens, densities_lst = calc_size_density(clusters)
##    mean_3d_dens, med_3d_dens, td_densities_lst = calc_3D_polygon_density(clusters)
##    mean_cv_dens, med_cv_dens, cv_densities_lst = calc_cv_size_density(clusters)

##    img_props_lst = [mean_loc_num, med_loc_num, mean_radius, med_radius,
##                     mean_2d_dens, med_2d_dens, mean_3d_dens, med_3d_dens]
    
##    clstr_props_lst = [list(vols_dict.values()), radii_lst, densities_lst, td_densities_lst]

##    img_cols = ['Mean Localisation Number', 'Median Localisation Number', 'Mean Radius', 'Median Radius', 'Mean 2D Density',
##                'Median 2D Density', 'Mean 3D Density', 'Median 3D Density']
    clstr_cols = ['Label', 'Number of Localisations', 'Volume', 'Radius', '2D Density', '3D Density', 'Cluster']

##    img_props_df = pd.DataFrame(img_props_lst, columns = img_cols)

    gen_list, mean_vol, med_vol, mean_locs, med_locs, mean_2d_dens, med_2d_dens, mean_radius, med_radius, mean_3d_dens, med_3d_dens, cluster_props_dict, noise = calc_all(clusters, d2_th, d3_th)
    
    img_props_df = pd.DataFrame()
    img_props_df['Mean Volume'] = [mean_vol]
    img_props_df['Median Volume'] = [med_vol]
    img_props_df['Mean Localisation Number'] = [mean_locs]
    img_props_df['Median Localisation Number'] = [med_locs]
    img_props_df['Mean Radius'] = [mean_radius]
    img_props_df['Median Radius'] = [med_radius]
    img_props_df['Mean 2D Density'] = [mean_2d_dens]
    img_props_df['Median 2D Density'] = [med_2d_dens]
    img_props_df['Mean 3D Density'] = [mean_3d_dens]
    img_props_df['Median 3D Density'] = [med_3d_dens]
    
##    clstr_props_df = pd.DataFrame(clstr_props_lst, columns = clstr_cols)
##    clstr_props_df = pd.DataFrame(gen_list, columns = clstr_cols)
    clstr_props_df = pd.DataFrame()

    clstr_props_df['Label'] = [item[0] for item in gen_list]
    clstr_props_df['Number of Localisations'] = [item[1] for item in gen_list]
    clstr_props_df['Volume'] = [item[2] for item in gen_list]
    clstr_props_df['Radius'] = [item[3] for item in gen_list]
    clstr_props_df['2D Density'] = [item[4] for item in gen_list]
    clstr_props_df['3D Density'] = [item[5] for item in gen_list]
    clstr_props_df['Length'] = [item[6] for item in gen_list]
    clstr_props_df['Width'] = [item[7] for item in gen_list]
    clstr_props_df['Cluster'] = [item[8] for item in gen_list]
    
##    print(clstr_props_df)
    
    return img_props_df, clstr_props_df, cluster_props_dict, noise

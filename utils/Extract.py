import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn import metrics
import plotly
import math
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scipy
import hdbscan
import json
import scipy
from scipy.spatial import distance_matrix, ConvexHull
import os
import statistics
import functools


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

##
##def calc_cv_size_density(clusters):
##    """
##    Calculates the mean and median densities of the xy convex hull containing each cluster.
##    """
##    densities_lst = []
##    sizes_lst = []
##    size_dens_3d_dict = {}
##    for cluster in clusters.values():
##        points = np.array([item[:3] for item in cluster])
##        hull = ConvexHull(points)
##        temp_vol = hull.volume
##        n = len(points)
##        temp_density = n / temp_vol
##        densities_lst.append(temp_density)
##        
##    mean_dens = statistics.mean(densities_lst)
##    median_dens = statistics.median(densities_lst)
##
##    return mean_dens, median_dens, densities_lst


def calc_all(clusters):
    """
    Calculates the volume of the convex hull containing each cluster.
    Param:
     ** no_noise_lst: labeled localizations which were assigned to clusters (type list).
    """
    gen_lst = []
    vols_list = []
    loc_list = []
    dens_2d_list = []
    dens_3d_list = []
    radii_list = []
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
    if len(clusters.values()) > 0:
        for label,cluster in clusters.items():

            points_3d = np.array([point[:3] for point in cluster])

            # Calculate the volume of current cluster
            volume = ConvexHull(points_3d).volume
            vols_list.append(volume)

            # Count the number of localisations in current clusster
            loc_num = len(points_3d)
            loc_list.append(loc_num)

            # Calculate the 2D density of current cluster
            points_2d = np.array([item[:2] for item in cluster])
            hull = ConvexHull(points_2d)
            corners = list(set(functools.reduce(lambda x,y: x+y,
                                                [[(a,b) for a,b in x] for x in points_2d[hull.simplices]])))
            temp_area = PolygonArea(PolygonSort(corners))
            density_2d = loc_num / temp_area
            dens_2d_list.append(density_2d)

            # Calculate 2D radius
            hull = ConvexHull(points_2d)
            perimeter = hull.area
            size = hull.volume
            radius = 2 * size / perimeter
            radii_list.append(radius)
            
            # Calculate 3D density
            hull = scipy.spatial.ConvexHull(points_3d)
            density_3d = (1000 * loc_num) / volume
            dens_3d_list.append(density_3d)

            clst_lst = [int(label), loc_num, volume, radius, density_2d, density_3d, cluster]
            gen_lst.append(clst_lst)
            cluster_props_dict[label] = clst_lst[1:6]
            
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
        print('No clusters were found!')

    return gen_lst, mean_vol, med_vol, mean_locs, med_locs, mean_dens_2d, med_dens_2d, mean_radius, med_radius, mean_dens_3d, med_dens_3d, cluster_props_dict


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

def extract_AP(clusters):
    """
    @Param:
    @@clusters - dict, keys are cluster labels, values are localisations assigned to cluster with label==key
    @Return:
    @@prop_lst - list, cluster properties of each cluster in the dataset.
    @@gen_lst - list, mean and median cluster properties of the image.
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

    gen_list, mean_vol, med_vol, mean_locs, med_locs, mean_2d_dens, med_2d_dens, mean_radius, med_radius, mean_3d_dens, med_3d_dens, cluster_props_dict = calc_all(clusters)
    
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
    clstr_props_df['Cluster'] = [item[6] for item in gen_list]
    
##    print(clstr_props_df)
    
    return img_props_df, clstr_props_df, cluster_props_dict
    
    
    


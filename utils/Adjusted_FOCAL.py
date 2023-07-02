import pandas as pd
import numpy as np
from collections import Counter
from sklearn import metrics
from sklearn.decomposition import PCA
import math
import ast
import scipy
from scipy.spatial import distance_matrix, ConvexHull
import statistics
import functools
from traceback import format_exc
import torch
import torch.utils.data as data
import time
import sys
import csv
import collections
import os

sys.setrecursionlimit(10**6)

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



class Grid:

    def __init__(self, pts_arr, voxel_dict, min_L, min_C, sig, minPC):
        """
        Define a new grid of size max_coords[0] x max_coords[1] x max_coords[2]
        divided into sections of size sig x sig x sig
        @ Params:
         @@ sig: 
         @@ pts_arr: [0] - x, [1] - y, [2] - z, [3] - photon-count
         @@ voxel_dict:
         @@ min_L:
         @@ min_C:
        """
        self.minC = min_C #minimum score of a group of voxels located next to each other required to call the group a cluster.
        self.minL = min_L #minimum score of localisation number in neighbouring voxels required in order to take this voxel into consideration.  
        self.sig = sig #sig^3 will be each voxel's volume.
        self.minPC = minPC #minimum average photon-count value per localisation required in order to call a group a cluster
        self.voxel_dict = dict()
        self.num_of_points = len(pts_arr)

        self.x_min = np.amin(pts_arr[:,0]) #minimum x coordinate of all points. 
        self.y_min = np.amin(pts_arr[:,1]) #minimum y coordinate of all points. 
        self.z_min = np.amin(pts_arr[:,2]) #minimum z coordinate of all points.

        self.x_max = np.amax(pts_arr[:,0]) #maximum x coordinate of all points. 
        self.y_max = np.amax(pts_arr[:,1]) #maximum y coordinate of all points.
        self.z_max = np.amax(pts_arr[:,2]) #maximum z coordinate of all points.
        
        #An array of all the sigma steps from *_min to *_max
        #(for instance: points: 100, 140, 203, 340, 400 --> x_edges: [100, 200, 300, 400])
        #Adding 1 to *_max will ensure that if max point is a multiple of sigma the edge *_max will also be in *_edges.
        self.x_edges = np.arange(self.x_min, self.x_max + 1, self.sig) 
        self.y_edges = np.arange(self.y_min, self.y_max + 1, self.sig)
        self.z_edges = np.arange(self.z_min, self.z_max + 1, self.sig)

        #The last value in *_edges (400 in ths last example) 
        self.last_x = self.x_edges[len(self.x_edges) - 1]
        self.last_y = self.y_edges[len(self.y_edges) - 1]
        self.last_z = self.z_edges[len(self.z_edges) - 1]
        
        voxel_vol = self.sig * self.sig * self.sig

    
    def assign_pts_to_voxels(self, pts_arr):
        """
        Goes over all points in the dataframe and assigns each point to a specific voxel in the grid.
        For instance the point [103,200,99] will be assigned to voxel [1, 2, 0].
        Params:
         ** pts_arr: all points in the dataset. 
        """
        pnt_lst = pts_arr.tolist()
        for p in pnt_lst:
            sig_x = math.floor(p[0] / self.sig)
            sig_y = math.floor(p[1] / self.sig)
            sig_z = math.floor(p[2] / self.sig)
            key = str([sig_x, sig_y, sig_z])

            if key not in self.voxel_dict: # Check if the voxel has already been created
                self.voxel_dict[key] = [[], [], [], [], True, [-1]] # If not, create it.
                """
                [0] = list: of points in the current voxel.
                [1] = int: Voxel's neighbours' score sum.
                [2] = bool: True if voxel is edge, else False.
                [3] = bool: True if voxel's score is above minL threshold, else False.
                [4] = bool: True if voxel's average photon-count is above pc, else False. set to True by default.
                [5] = int: FOCAL cluster label
                """
            self.voxel_dict[key][0].append(p) # Add point to relevant voxel 


    def is_edge_voxel(self, voxel_coords_lst):
        """
        Checks whether a voxel is on the edge of the grid.
        Marks True if the voxel is on the edge of the grid or False if it is interior.
        Params:
         ** voxel_coords_lst: coordinates of a single voxel (type list).
        Ret:
         ** True if the voxel is on the edge.
         ** False if the voxel is in the interior of the grid.
        """
        #array of the coordinates that are the min edges of the grid.
        min_coords_voxel = [self.x_min / self.sig, self.y_min / self.sig, self.z_min / self.sig]
        #array of the coordinates that are the max edges of the grid.
        max_coords_voxel = [self.last_x / self.sig, self.last_y / self.sig, self.last_z / self.sig] 
        #if at least one of coordinates of the voxel equals to the corresponding min/max coordinates, then this is an edge voxel. 
        for i in range(3):
            if voxel_coords_lst[i] == min_coords_voxel[i]:
                return True
            elif voxel_coords_lst[i] == max_coords_voxel[i]:
                return True
        return False

    def neigh_score_sum(self):
        """
        Labels the voxel as edge/interior, calculates the sum of each interior voxel's neighbours' scores (plus its own score)
        and sets its score to that sum.
        """
        for key in self.voxel_dict.keys():
            if len(self.voxel_dict[key][0])>0:
                score = 0
                voxel_coords_lst = ast.literal_eval(key)
                bool_edge_voxel = self.is_edge_voxel(voxel_coords_lst)
                #Label the voxel as edge/interior
                self.voxel_dict[key][2] = bool_edge_voxel
                if bool_edge_voxel == False: #Calculating score only if this is an interior voxel.
                    left_down_forward_voxel = [voxel_coords_lst[0] - 1, voxel_coords_lst[1] - 1, voxel_coords_lst[2] - 1]  
                    #Take all the 3*3*3-1 voxels surrounding this voxel 
                    for i in range(0, 3):
                        for j in range(0, 3):
                            for k in range(0,3):
                                neigh_key = str([left_down_forward_voxel[0] + i, left_down_forward_voxel[1] + j, left_down_forward_voxel[2] + k])
                                if neigh_key in self.voxel_dict:
                                    score = score + len(self.voxel_dict[neigh_key][0])
                #Set the voxel's score to the sum of scores.
                self.voxel_dict[key][1].append(score) 
            else:
                #Set the voxel's score to the sum of scores.
                self.voxel_dict[key][1].append(0)

    def change_edge_score(self):
        """
        Sets all edge voxels' scores to 0
        """
        for key in self.voxel_dict.keys():
            if self.voxel_dict[key][2] == True:
                self.voxel_dict[key][1] = [0]

    def is_above_threshold(self):
        """
        Checks whether each voxel's neighbours' scores (plus its own score) is above minL.
        minL - the density threshold set by the user.
        """
        for key in self.voxel_dict.keys():
            if self.voxel_dict[key][1][0] < self.minL:
                self.voxel_dict[key][3] = False #This voxel's total score is below minL threshold
            else:
                self.voxel_dict[key][3] = True #This voxel's total score equals to or bigger than minL threshold

    def is_above_pc(self):
        """
        Calculates each voxel's average photon-count if pc threshold has been set by the user.
        """
        cnt = 0
        if self.minPC != -1:
            for key,val in self.voxel_dict.items():
                tmp_pc_lst = [pc[3] for pc in val[0]]
                l = len(tmp_pc_lst)
                total_pc = sum(tmp_pc_lst)
                avg_pc = total_pc / l
                if (avg_pc < self.minPC):
                    self.voxel_dict[key][4] = False
                    cnt += 1
        print(cnt, ' Clusters have been dropped due to low photon-count')
            

    def is_FOCAL_cluster(self, pts_df):
        """
        Checks whether a group of voxels constitute of a cluster according to FOCAL's parameters.
        Param:
         ** pts_df: all points in the dataset (type pandas dataframe).
        Ret:
         ** full_lst: all points with labels (type list)
         ** no_noise_lst: only points assigned to clusters with labels (type list)
        """
        n = len(pts_df.index)
        labels = [-1] * n
        pts_df['Labels'] = labels
        vox_dict = self.voxel_dict.copy()
        label = 0
        for key in vox_dict.keys():
            c_score = 0 #The number of core voxels in a cluster candidate.
            val = vox_dict[key]
            cluster_set = set() #An empty set of voxels.
            if val[2] == False: #If the voxel is not an edge voxel.
                if val[3] == True: #If this voxel's total score >= minL (density threshold).
                    if val[5] == [-1]: #If this voxel wasn't assigned to any cluster yet. 
                        self.rec_vox(key, val, vox_dict, cluster_set)
                        for v in cluster_set: #v is a voxel's key for the dictionary.
                            if self.voxel_dict[v][3] == True: #If v is the key of a core voxel.
                                c_score += 1 #The final value of c_score will determine whether the group of voxels is a cluster.
            if c_score >= self.minC: #If the number of core voxels in the group is greater than or equal to minC -> it is a cluster.
                for v in cluster_set:
                    self.voxel_dict[v][5] = [label] #All voxels in the cluster get the same cluster label.
                label += 1 #The label is incremented for the next cluster to be found.
        self.labels = list(range(0, label)) #range is a half open interval so it goes up to (label - 1) as required.
        self.cluster_num = label
        full_lst = []
        for v in (self.voxel_dict).values(): #Add each localization, with its cluster label to a list of all localizations.
            if (len(v[0]) > 0):
                for p in v[0]:
                    p1 = p
                    p1.append(v[5][0]) #Each element in the list is of the form (x,y,z,pc,label)
                    full_lst.append(p1)
        no_noise_lst = []
        for p in full_lst: #Add all clustered localizations to a sub-list.
            if p[4] != -1:
                no_noise_lst.append(p)
        return full_lst, no_noise_lst

    def rec_vox(self, voxel_key, voxel_val, voxel_dict, cluster_set):
        """
        Recursively calculates a voxel's neighbours' score sum.
        Recieves only voxels with [2] == False (Not an edge voxel) and [3] == True (surpassed minL threshold).
        Params:
         ** voxel_key: the id of a voxel (type str).
         ** voxel_val: the values assigned to the voxel (type list of lists).
         ** voxel_dict: the dictionary containing all voxels in the grid.
         ** cluster_set: all voxels which belong to this cluster (type set).
        """
        neighbours = []
        vox_coords = ast.literal_eval(voxel_key)
        ctr = 1
        if voxel_key not in cluster_set:
            if len(voxel_val[0]) > 0:
                cluster_set.add(voxel_key)
                if voxel_val[3] == True: #If this voxel's total score is equal to or larger than minL.
                    #print("\nVoxel's Neighbours' Score Sum: ", voxel_val[1], "\n")
                    if voxel_val[4] == True: #If this voxel's average photon-count >= minPC.
                        if voxel_val[2] == False: #If the voxel is not an edge voxel.
                            neighbours.append(str([vox_coords[0], vox_coords[1], vox_coords[2] + 1]))
                            neighbours.append(str([vox_coords[0], vox_coords[1], vox_coords[2] - 1]))
                            neighbours.append(str([vox_coords[0], vox_coords[1] + 1, vox_coords[2]]))
                            neighbours.append(str([vox_coords[0], vox_coords[1] - 1, vox_coords[2]]))
                            neighbours.append(str([vox_coords[0] + 1, vox_coords[1], vox_coords[2]]))
                            neighbours.append(str([vox_coords[0] - 1, vox_coords[1], vox_coords[2]]))
                            for n in neighbours:
                                if n not in cluster_set:
                                    if n in voxel_dict:
                                        self.rec_vox(n, voxel_dict[n], voxel_dict, cluster_set)
                                        ctr += 1
        

    def get_axis_reduction_pca(self, np_array):
        """
        Reduces the data's dimension from 3D to 2D.
        Param:
         ** np_array: points in a cluster (type numpy array).
        Ret:
         ** reduced_data: the input points reduced to 2D.
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

    def div_lst_to_clusters(self, full_lst, no_noise_lst):
        """
        Divides list of labeled points into a dictionary.
        The dictionary's keys represent cluster labels.
        The dictionary's values represent points belonging to the cluster with relevant label = key.
        Param:
         ** no_noise_lst: labeled localizations which were assigned to clusters (type list).
        """
        clstrs_dict = {}
        self.num_of_locs_assigned_to_clusters = len(no_noise_lst)
##        print("Number of clusters before size screening: ", self.cluster_num)
##        print('Number of points assigned to clusters: ', self.num_of_locs_assigned_to_clusters)
        if self.cluster_num > 0:
            for i in range(self.cluster_num):
                label = str(i)
                clstrs_dict[label] = []
            for p in no_noise_lst:
                label = p[4]
                clstrs_dict[str(label)].append(p)
            tmp_dict = clstrs_dict.copy()

        self.cluster_num = len(self.labels)
        self.clusters = clstrs_dict
##        print(self.clusters.keys())
        return full_lst, no_noise_lst

    def calc_volumes(self):
        """
        Calculates the volume of the convex hull containing each cluster.
        Param:
         ** no_noise_lst: labeled localizations which were assigned to clusters (type list).
        """
        vols_dict = {}
        for cluster in self.clusters.values():
            label = str(cluster[0][4])
            clstr = []
            for p in cluster:
                p1 = p[0:3]
                clstr.append(p1)
            points = np.array(clstr)
            volume = ConvexHull(points).volume
            vols_dict[label] = volume
        self.clstr_vols = vols_dict

    def calc_localizations(self):
        """
        Calculates the mean and median number of localizations per cluster.
        """
        pnts_lst = []
        for cluster in self.clusters.values():
            pnts_lst.append(len(cluster))
        median_loc = statistics.median(pnts_lst)
        mean_loc = statistics.mean(pnts_lst)
        self.mean_loc = mean_loc
        self.median_loc = median_loc

    def calc_3D_polygon_density(self):
        """
        Calculates the mean and median densities of the 3D convex hull containing each cluster.
        """
        densities_lst = []
        for cluster_lst in self.clusters.values():
            points = np.array([point[:3] for point in cluster_lst])
            hull = scipy.spatial.ConvexHull(points)
            vol = hull.volume
            n = len(cluster_lst)
            temp_density = (1000 * n) / vol
            densities_lst.append(temp_density)
        mean_dens = statistics.mean(densities_lst)
        median_dens = statistics.median(densities_lst)
        self.mean_density = mean_dens
        self.median_density = median_dens
        self.densities = densities_lst

    def calc_red_poly_density_size(self):
        """
        Calculates the mean and median sizes and densities of reduced convex hull containing each cluster.
        """
        reduced_poly_size = []
        reduced_poly_density = []
        reduced_size_density_dict = {}
        for cluster in self.clusters.values():
            points = np.array([item[:3] for item in cluster])
            reduced_cluster = self.get_axis_reduction_pca(points)
            if reduced_cluster is not None:
                reduced_convex_hull = ConvexHull(reduced_cluster)
                corners = list(set(functools.reduce(lambda x,y: x + y,
                                                    [[(a.tolist()[0][0], a.tolist()[0][1]) for a in x]
                                                     for x in reduced_cluster[reduced_convex_hull.simplices]])))
                temp_poly_size = PolygonArea(PolygonSort(corners))
                reduced_poly_size.append(temp_poly_size)
                temp_density = float((1000 * len(reduced_cluster)) / temp_poly_size)
                reduced_poly_density.append(temp_density)
                reduced_size_density_dict[str(cluster[0][3])] = [temp_poly_size, temp_density]
                
        mean_size = statistics.mean(reduced_poly_size)
        mean_dens = statistics.mean(reduced_poly_density)
        median_size = statistics.median(reduced_poly_size)
        median_dens = statistics.median(reduced_poly_density)
        self.mean_reduced_polygon_size = mean_size
        self.mean_reduced_polygon_density = mean_dens
        self.median_reduced_polygon_size = median_size
        self.median_reduced_polygon_density = median_dens
        self.reduced_polygon_size = reduced_poly_size
        self.reduced_polygon_density = reduced_poly_density
        return reduced_size_density_dict


    def calc_poly_surrounding_flat_cluster_size(self):
        """
        Calculates the mean and median size of the surrounding flat polygon of each cluster.
        """
        sizes_lst = []
        for cluster in self.clusters.values():
            points = np.array([item[:2] for item in cluster])
            hull = ConvexHull(points)
            temp_area = hull.area
            sizes_lst.append(temp_area)
        mean_size = statistics.mean(sizes_lst)
        median_size = statistics.median(sizes_lst)
        self.flat_poly_mean_size = mean_size
        self.flat_poly_median_size = median_size

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
        if len(sizes_lst) > 0: 
            mean_size = statistics.mean(sizes_lst)
            median_size = statistics.median(sizes_lst)
        else:
            mean_size = -1
            median_size = -1
        self.flat_poly_mean_size = mean_size
        self.flat_poly_median_size = median_size
    
    def calc_cv_size_density(self):
        """
        Calculates the mean and median densities of the xy convex hull containing each cluster.
        """
        densities_lst = []
        sizes_lst = []
        self.size_dens_3d_dict = {}
        for cluster in self.clusters.values():
            points = np.array([item[:3] for item in cluster])
            hull = ConvexHull(points)
            temp_vol = hull.volume
            n = len(points)
            temp_density = n / temp_vol
            densities_lst.append(temp_density)
            self.size_dens_3d_dict[str(cluster[0][3])] = [temp_vol, temp_density]
        mean_dens = statistics.mean(densities_lst)
        median_dens = statistics.median(densities_lst)
        self.cv_mean_density = mean_dens
        self.cv_median_density = median_dens
        self.cv_densities = densities_lst    

    def calc_polygon_size_density(self):
        """
        Calculates the mean and median densities of the xy convex hull containing each cluster.
        """
        densities_lst = []
        sizes_lst = []
        self.size_dens_dict = {}
        for cluster in self.clusters.values():
            points = np.array([item[:2] for item in cluster])
            hull = ConvexHull(points)
            corners = list(set(functools.reduce(lambda x,y: x+y,
                                                [[(a,b) for a,b in x] for x in points[hull.simplices]])))
            temp_area = PolygonArea(PolygonSort(corners))
            n = len(cluster)
            temp_density = n / temp_area
            densities_lst.append(temp_density)
            self.size_dens_dict[str(cluster[0][3])] = [temp_area, temp_density]
        mean_dens = statistics.mean(densities_lst)
        median_dens = statistics.median(densities_lst)
        self.poly_mean_density = mean_dens
        self.poly_median_density = median_dens
        self.poly_densities = densities_lst

    def calc_polygon_radius(self):
        radii = []
        for cluster in self.clusters.values():
            points = np.array([item[:2] for item in cluster])
            hull = ConvexHull(points)
            perimeter = hull.area
            size = hull.volume
            radius = 2 * size / perimeter
            radii.append(radius)
            self.size_dens_dict[str(cluster[0][3])].append(radius)
        self.mean_radius = statistics.mean(radii)
        self.median_radius = statistics.median(radii)

    def drop_density_threshold(self, no_noise_lst, full_lst):
        clstrs_dict = self.clusters.copy()
        for cluster in self.clusters.values():
            key = str(cluster[0][3])
            dens = self.size_dens_dict[key][1]
            if dens < 0.0005:
                for p in cluster:
                    if p[4] == key:
                        p[4] = -1
                        full_lst.append(p)
                        no_noise_lst.remove(p)
                del clstrs_dict[key]
                print("\nCluster with label ", key, " dropped because its density = ", dens, " < 0.0005")
                self.labels.remove(ast.literal_eval(key))
        self.cluster_num = len(self.labels)
        self.clusters = clstrs_dict


def create_FOCAL_grid(pts_arr, minC, minL, sigma, minPC = -1):
    grid = Grid(pts_arr, None, minL, minC, sigma, minPC = -1)
    print("Generated grid")
    grid.assign_pts_to_voxels(pts_arr)
    print("Assigned points to voxels")
    grid.neigh_score_sum()
    print("Set neighbours' score sums")
    grid.change_edge_score()
    print("Changed edge scores")
    return grid

def enforce_FOCAL_thresholds(pts_df, grid, minC, minL, boo):
    grid.minC = minC
    grid.minL = minL
    if boo == 0:
        print("\n******************* Random FOCAL *******************\n")
    else:
        print("\n******************* Actual FOCAL *******************\n")
    grid.is_above_threshold()
    print("Identified above threshold voxels")
    grid.is_above_pc()
    full_lst, no_noise_lst = grid.is_FOCAL_cluster(pts_df)
    print("Identified FOCAL clusters")
    cluster_num = grid.cluster_num
    return cluster_num

def param_scan(pts_df, pts_arr, minL_min, minL_max, minL_step,
               minC_min, minC_max, minC_step,
               grid_min, grid_max, grid_step):

    x_min = np.amin(pts_arr[:,0])
    x_max = np.amax(pts_arr[:,0])
    y_min = np.amin(pts_arr[:,1])
    y_max = np.amax(pts_arr[:,1])
    z_min = np.amin(pts_arr[:,2])
    z_max = np.amax(pts_arr[:,2])
    # A randomly scattered localizations table
    # Within the same volume as the original data.
    minL_rnd = np.random.uniform((x_min,y_min,z_min,0,0),(x_max,y_max,z_max,0,0),
                                 (len(pts_arr),5))
    minL_range = np.arange(minL_min, minL_max + 1, minL_step)
    minC_range = np.arange(minC_min, minC_max + 1, minC_step)
    sig_range = np.arange(grid_min, grid_max + 1, grid_step)
    print('\n******************** New Params ********************\n',
          "minL range: ", list(minL_range),
          "\nminC range: ", list(minC_range),
          "\nGrid Size range: ", list(sig_range), "\n")
    opt_minLs = []
    
    for sig_ind,sig_val in enumerate(sig_range):
        for c_ind,c_val in enumerate(minC_range):
            cluster_num_minL = (-1) * np.ones((minL_range.size, 1))
            
            for l_ind,l_val in enumerate(minL_range):
                print("\nSIGMA: ", sig_val, "\nminC: ", c_val, "\nminL: ", l_val, "\n")
                minL_found = False
                
                try: #To find the optimal minL value
                    rand_grid = create_FOCAL_grid(minL_rnd, -1, -1, sig_val)
                    minL_cluster_num = enforce_FOCAL_thresholds(pts_df, rand_grid, c_val, l_val, 0)
                    print("\nRandom cluster number: ", minL_cluster_num, "\n")
                    
                    if minL_cluster_num == 0:
                        minL_found = True
                        act_grid = create_FOCAL_grid(pts_arr, -1, -1, sig_val)
                        clusters = enforce_FOCAL_thresholds(pts_df, act_grid, c_val, l_val, 1)
                        print("\nActual cluster number: ", clusters, "\n")
                        opt_minLs.append(np.array([sig_val, c_val, l_val, clusters]))
                        break
                    
                except Exception as e:
                    print(e)
                    print("FOCAL error at minL = {}. minC = {}. GridSize = {}".format(sig_val,c_val,l_val))
                    continue
                
                finally:
                    if minL_found:
                        break

    opt_minLs_arr = np.array(opt_minLs)
    sigmas = [sig[0] for sig in opt_minLs]
    minCs = [C[1] for C in opt_minLs ]
    minLs = [L[2] for L in opt_minLs]
    cluster_nums = [c_num[3] for c_num in opt_minLs]

    opt_minL_df = pd.DataFrame()
    opt_minL_df['Grid Size'] = sigmas
    opt_minL_df['minC'] = minCs
    opt_minL_df['Optimal minL'] = minLs
    opt_minL_df['Number of Clusters found'] = cluster_nums
    
    return opt_minL_df
    

def FOCAL(xyz_df, minL, minC, sigma, minPC):
    """
    Main Function for FOCAL Algorithm - calls all sub-methods
    @ Params:
     ** xyz_df - dataframem, columns = ['x', 'y', 'z', 'photon-count']
     ** minL - threshold for the minimum number of localizations in a single bin's neighbour score - set by the user.
     ** minC - threshold for the minimum number of bins that constitute a cluster - set by the user.
     ** Sigma - sigma ** 3 is the size of the grid used by FOCAL - set by the user.
     ** minPC - photon-count threshold.
    """
    xyz = xyz_df.to_numpy()
    grid = Grid(xyz, None, minL, minC, sigma, minPC)
    grid.assign_pts_to_voxels(xyz)
    grid.neigh_score_sum()
    grid.change_edge_score()
    grid.is_above_threshold()
    grid.is_above_pc()
    full_lst, no_noise_lst = grid.is_FOCAL_cluster(xyz_df)
    full_lst1, no_noise_lst1 = grid.div_lst_to_clusters(full_lst, no_noise_lst)

##    all_locs_df = pd.DataFrame(full_lst1, columns = ['x', 'y', 'z', 'Label'])
##    clustered_df = pd.DataFrame(no_noise_lst1, columns = ['x', 'y', 'z', 'Label'])
    
    all_locs_df = pd.DataFrame()
    all_locs_df['x'] = [p[0] for p in full_lst1]
    all_locs_df['y'] = [p[1] for p in full_lst1]
    all_locs_df['z'] = [p[2] for p in full_lst1]
    all_locs_df['Label'] = [p[4] for p in full_lst1]
    
    clustered_df = pd.DataFrame()
    clustered_df['x'] = [p[0] for p in no_noise_lst1]
    clustered_df['y'] = [p[1] for p in no_noise_lst1]
    clustered_df['z'] = [p[2] for p in no_noise_lst1]
    clustered_df['Label'] = [p[4] for p in no_noise_lst1]

    return all_locs_df, clustered_df


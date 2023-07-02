import numpy as np
import pandas as pd
import os
import torch.utils.data as data

from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from traceback import format_exc
from pandas.io.parsers import read_csv
import statistics
from utils.Cluster_Identification_Algorithms import sample_and_group, dbscan_cluster_and_group,\
     hdbscan_cluster_and_group, focal_cluster_and_group
from utils.Extract import extract_AP
from collections import Counter
from datetime import datetime
from pathlib import Path
from utils.Plot import plot_res


class dstorm_dataset(data.Dataset):
    
    def __init__(self, input_path, output_path, selected_files, algs, config, pbar, workers = 8):
        '''
        @params:
            input_path - str, path to file/files/directory
            output_path - str, path to desired output directory
            algs - list of bools, [DBSCAN, HDBSCAN, FOCAL]
            config - list, configuration parameters for each algorithm set to TRUE in algs
            pw - PlotWindow, empty window to show results
            pbar - progress bar
        '''
        fileslist = []
        self.input_path = input_path[1][0]
        self.pc_th = 0.0
        self.px_th = 1000.0
        self.clust_res = {}
        self.error = None

        for alg in algs:
            if alg[1] == True:
                self.pc_th = (config.get(str(alg[0])))[0] # Set the minimum photon-count threshold
                self.px_th = (config.get(str(alg[0])))[1] # Set the maximum x-precision threshold

        if input_path[0] == 'files':
            if isinstance(self.input_path, list):
                for full_path in self.input_path:
                    for sf in selected_files:
                        if sf in full_path:
                            fileslist.append(full_path)
        elif input_path[0] == 'dir':
            if isinstance(self.input_path, str) and os.path.isdir(self.input_path):
                for filename in os.listdir(self.input_path):
                    if filename.endswith('.csv'):
                        if filename in selected_files:
                            full_path = os.path.join(self.input_path, filename)
                            fileslist.append(full_path)
        else:
##            raise ValueError(f'{self.path} input path is NOT a path to a directory or .csv file')
            self.error = "The input path is NOT a path to a directory or .csv file!"

        pool = Pool(workers)
        
        try:
            df_rows = []
            with tqdm(total = len(fileslist)) as progbar:
                result = pool.imap_unordered(self.parse_row, fileslist) # Load files
                for r in result:
                    df_rows.append(r)
                    ##pbar.update(1)
        finally:
            pool.close()
            pool.terminate()

        self.orig_df = pd.DataFrame(df_rows)
        
        if self.orig_df is not None:
            self.data = self.orig_df.to_dict(orient = 'records')
            self.indices = self.orig_df.index.values
            self.index_to_row = {i: dpoint for (i, dpoint) in zip(self.indices, self.data)}
        else:
            self.data = None
            self.indices = None
            self.index_to_row = None
        for i,alg in enumerate(algs):
            if alg[1] == True:
                params = config.get(str(alg[0]))
                if alg[0] == 'DBSCAN':
                    db = DBSCAN_dataset(self.data, params, output_path)
                    self.clust_res[alg[0]] = db.res
                elif alg[0] == 'HDBSCAN':
                    hdb = HDBSCAN_dataset(self.data, params, output_path)
                    self.clust_res[alg[0]] = hdb.res
                elif alg[0] == 'FOCAL':
                    foc = FOCAL_dataset(self.data, params, output_path)
                    self.clust_res[alg[0]] = foc.res
##                pbar.setValue(i)

        
    def parse_path(self, path):
        """
        Change to parse additional/other fields from full path
        """
        parsed_path = {'filename': os.path.basename(path)}
        return parsed_path
    

    def process_pointcloud_df(self, pointcloud):
        """
        Manipulate pointcloud DataFrame
        
        @param pointcloud: dataframe, read from input file.csv
        **********************************************
        @ret df_row: dict, with one dataframe value and one int value
        """
        self.orig_loc_num = len(pointcloud.index)
        print("Dropping localisations due to photon intensity filter (%f)" % self.pc_th)
        tmp_pointcloud = pointcloud.loc[pointcloud['photon-count'] >= self.pc_th]

        print("Dropping localisations due to x-precision filter (%f)" % self.px_th)
        fin_pointcloud = tmp_pointcloud.loc[tmp_pointcloud['precisionx'] <= self.px_th]
##        pc = fin_pointcloud[['x', 'y', 'z']]
        pc = fin_pointcloud[['x', 'y', 'z', 'photon-count']]
        df_row = {'pointcloud': pc, 'num_of_points': len(pc)}
        return df_row


    def parse_row(self, path):
        """
        Row parser -
            args (list): list of all arguments the function is getting.
                            passing arguments this way is necessary for pool.imap_unordered
        @param path: str, path to input file, file name, and pointcloud
        ***********************************************************************************
        @ret row: dict, containing path, 
        """
        row = {'path': path}
        cols = ['photon-count','x', 'y', 'z', 'precisionx']
        try:
            parsed_path = self.parse_path(row['path'])

            with open(row['path'], 'r') as f:
                processed_pointcloud = self.process_pointcloud_df(read_csv(f, usecols = cols))
            row = {**row, **parsed_path, **processed_pointcloud}

        except Exception as e:
            row['Exception'] = format_exc()

        return row

# ************************************************ Sub Classes of Cluster ID Algorithms ************************************************ #

class DBSCAN_dataset(dstorm_dataset):

    def __init__(self, data, params, output_path):
        '''
        Creates an object of 3D localisations, each with a cluster label given by DBSCAN
        @param data: 
        @param params:
        '''
        self.alg = 'DBSCAN'
        self.min_npoints = params[4]
        self.epsilon = params[5]
        self.min_cluster_points = params[6]
        self.output_path = output_path
        ### if the user didn't set a PCA standard deviation
        if len(params) <= 7:
            self.pca_stddev = 1.0
        ### if the user did set a PCA standard deviation
        else:
            self.pca_stddev = params[7]
        self.res = {}
        for i,val in enumerate(data):
            self.call_DBSCAN(val['filename'], val['pointcloud'])


    def call_DBSCAN(self, filename, scanned_data):
        """
        Calls DBSCAN and creates self.res - the resulting array of localisations with cluster labels
        @param filename: the filename of the data to run DBSCAN on
        @param scanned_data: the localisations to run DBSCAN on
        """
##        clustered_df, xyzl_df, params, number_of_locs, number_of_locs_assigned_to_clusters = dbscan_cluster_and_group(xyz = scanned_data,
##                                                     min_npoints = self.min_npoints,
##                                                     eps = self.epsilon,
##                                                     min_cluster_points = self.min_cluster_points)
        xyz_df = scanned_data.copy()
        img_props, cluster_props, cluster_props_dict, xyzl = dbscan_cluster_and_group(xyz = xyz_df,
                                                                                      min_npoints = self.min_npoints,
                                                                                      eps = self.epsilon,
                                                                                      min_cluster_points = self.min_cluster_points,
                                                                                      fname = filename)
        if len(img_props.index) > 0:
            
            new_fn = 'DBSCAN_' + str(self.min_npoints) + '_' + str(self.epsilon)

            fname = str(Path(filename).stem)

    ##        plt_main(xyzl, filename, 'DBSCAN', self.output_path)

            now = datetime.now() # datetime object containing current date and time
##            dt_string = now.strftime("%Y.%m.%d %H:%M:%S") # YY/mm/dd H:M:S
            dt_string = now.strftime("%Y/%m/%d %H:%M:%S") # YY/mm/dd H:M:S

            plot_res(xyzl, cluster_props_dict, fname, 'DBSCAN', self.output_path)
            
            img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
            cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'

            img_path = os.path.join(self.output_path, img_props_name)
            cluster_path = os.path.join(self.output_path, cluster_props_name)
            img_props.to_excel(self.output_path + '/' + img_props_name)
            cluster_props.to_excel(self.output_path + '/' + cluster_props_name)
        else:
            print('No Clusters were found in DBSCAN!')
        


class HDBSCAN_dataset(dstorm_dataset):

    def __init__(self, data, params, output_path):
        
        self.alg = 'HDBSCAN'
        self.min_cluster_points = params[4]
        self.epsilon_threshold = params[5]
        self.min_samples = params[6]
        self.extracting_alg = params[7]
        self.alpha = params[8]
        self.output_path = output_path
        ### if the user didn't set a PCA standard deviation
        if len(params) <= 9:
            self.pca_stddev = 1.0
        ### if the user did set a PCA standard deviation
        else:
            self.pca_stddev = params[9]
        self.res = []
        for i,val in enumerate(data):
            self.call_HDBSCAN(val['filename'], val['pointcloud'])

    '''
    def call_HDBSCAN(self, filename, scanned_data):
        xyzl = hdbscan_cluster_and_group(
            xyz = self.scanned_data,
            min_cluster_points = self.min_cluster_points,
            epsilon_threshold = self.epsilon_threshold,
            min_samples = self.min_samples,
            extracting_alg = self.extracting_alg,
            alpha = self.alpha
            )
        df, number_of_localizations, number_of_locs_assigned_to_clusters = extract_AP(labeled_pts = xyzl, pca_stddev = self.pca_stddev)
        new_fn = 'HDBSCAN_' + filename
        #self.res.append({new_fn: [df, number_of_localizations, number_of_locs_assigned_to_clusters]})
        self.res[new_fn] = [df, number_of_localizations, number_of_locs_assigned_to_clusters]
    '''


    def call_HDBSCAN(self, filename, scanned_data):
        """
        Calls HDBSCAN and creates self.res - the resulting array of localisations with cluster labels
        @param filename: the filename of the data to run HDBSCAN on
        @param scanned_data: the localisations to run HDBSCAN on
        """

        xyz_df = scanned_data.copy()
        img_props, cluster_props, cluster_props_dict, xyzl = hdbscan_cluster_and_group(
            xyz = xyz_df,
            min_cluster_points = self.min_cluster_points,
            epsilon_threshold = self.epsilon_threshold,
            min_samples = self.min_samples,
            extracting_alg = self.extracting_alg,
            alpha = self.alpha,
            fname = filename
            )

        if len(img_props.index) > 0:
            new_fn = 'HDBSCAN_' + str(self.epsilon_threshold)
            fname = str(Path(filename).stem)

    ##        plt_main(xyzl, filename, 'HDBSCAN', self.output_path)
            
            now = datetime.now() # datetime object containing current date and time
            dt_string = now.strftime("%Y.%m.%d %H{c}%M{c}%S").format(c = ':') # YY/mm/dd H:M:S
##            dt_string = now.strftime("%Y.%m.%d %H:%M:%S") # YY/mm/dd H:M:S

            plot_res(xyzl, cluster_props_dict, fname, 'HDBSCAN', self.output_path)

            img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
            cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'
            img_path = os.path.join(self.output_path, img_props_name)
            cluster_path = os.path.join(self.output_path, cluster_props_name)
            img_props.to_excel(self.output_path + '/' + img_props_name)
            cluster_props.to_excel(self.output_path + '/' + cluster_props_name)
        else:
            print('No clusters were found in HDBSCAN!')
        



class FOCAL_dataset(dstorm_dataset):

    def __init__(self, data, params, output_path):
        
        self.alg = 'FOCAL'
        self.sigma = params[4]
        self.minL = params[5]
        self.minC = params[6]
        self.minPC = params[7]
        print('self.minPC = ', self.minPC)
        print('self.sigma = ', self.sigma)
        self.output_path = output_path
        ### if the user didn't set a PCA standard deviation
        if len(params) <= 8:
            self.pca_stddev = 1.0
        ### if the user did set a PCA standard deviation
        else:
            self.pca_stddev = params[8]
        self.res = []
        for i,val in enumerate(data):
            self.call_FOCAL(val['filename'], val['pointcloud'])

    def call_FOCAL(self, filename, scanned_data):

        xyz_df = scanned_data.copy()
        img_props, cluster_props, cluster_props_dict, xyzl = focal_cluster_and_group(
            xyz = xyz_df,
            sigma = self.sigma,
            minL = self.minL,
            minC = self.minC,
            minPC = self.minPC,
            fname = filename
            )

        if len(img_props.index) > 0:
            new_fn = 'FOCAL_' + str(self.sigma) + '_' + str(self.minL) + '_' + str(self.minC)
            fname = str(Path(filename).stem)
            
            now = datetime.now() # datetime object containing current date and time
##            dt_string = now.strftime("%Y.%m.%d %H{c}%M{c}%S").format(c = ':') # YY/mm/dd H:M:S
##            dt_string = (str(now.year) + '.' + str(now.month) + '.' + str(now.day) + ' ' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)) # YY/mm/dd H:M:S
            dt_string = now.strftime("%Y.%m.%d %H_%M_%S") # YY/mm/dd H:M:S

            plot_res(xyzl, cluster_props_dict, fname, 'FOCAL', self.output_path, dt_string)
            
            img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
            cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'
            img_path = os.path.join(self.output_path, img_props_name)
            cluster_path = os.path.join(self.output_path, cluster_props_name)
        
            img_props.to_excel(self.output_path + '/' + img_props_name)
            cluster_props.to_excel(self.output_path + '/' + cluster_props_name)

        else:
            print('No clusters were found in FOCAL!')
        

import numpy as np
import pandas as pd
import os
import torch.utils.data as data
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from traceback import format_exc
from pandas.io.parsers import read_csv

from utils.Cluster_Identification_Algorithms import sample_and_group, dbscan_cluster_and_group,\
     hdbscan_cluster_and_group, focal_cluster_and_group

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

class dstorm_dataset(data.Dataset):
    
    def __init__(self, path, algs, config, workers = 8):
        fileslist = []
        self.path = path[1][0]
        self.res = {'DBSCAN': [], 'HDBSCAN': [], 'FOCAL': []}
        self.pc_th = 0.0
        for alg in algs:
            if alg[1] == True:
                self.pc_th = (config.get(str(alg[0])))[0]
        if path[0] == 'file':
            if isinstance(self.path, str) and os.path.isfile(self.path) and os.path.splitext(self.path)[1] == ".csv":
                fileslist.append(self.path)
                print('\nFILESLIST: ', fileslist)
        elif path[0] == 'files':
            if isinstance(self.path, list):
                for full_path in self.path:
                    fileslist.append(full_path)
        elif path[0] == 'dir':
            if isinstance(self.path, str) and os.path.isdir(self.path):
                for filename in os.listdir(self.path):
                    if filename.endswith('.csv'):
                        full_path = os.path.join(self.path, filename)
                        fileslist.append(full_path)
        else:
            raise ValueError(f'{self.path} input path is NOT a path to a directory or .csv file')

        pool = Pool(workers)
        
        try:
            df_rows = []
            with tqdm(total = len(fileslist)) as pbar:
                result = pool.imap_unordered(self.parse_row, fileslist)
                for r in result:
                    print(type(r))
                    df_rows.append(r)
                    pbar.update(1)
        finally:
            pool.close()
            pool.terminate()

        self.orig_df = pd.DataFrame(df_rows)
        print('Orig df:\n', self.orig_df, '\n')
        
        if self.orig_df is not None:
            self.data = self.orig_df.to_dict(orient = 'records')
            #print('Data Length: ', len(self.data), '\nData:\n', self.data)
            #print('\nTYPE: ', type(self.data[0]['pointcloud']), '\nDATA:\n', self.data[0]['pointcloud'])
            self.indices = self.orig_df.index.values
            self.index_to_row = {i: dpoint for (i, dpoint) in zip(self.indices, self.data)}
        else:
            self.data = None
            self.indices = None
            self.index_to_row = None

        for alg in algs:
            if alg[1] == True:
                params = config.get(str(alg[0]))
                if alg[0] == 'DBSCAN':
                    db = DBSCAN_dataset(self.data, params)
                    self.res[alg[0]] = db.res
                elif alg[0] == 'HDBSCAN':
                    hdb = HDBSCAN_dataset(self.data, params)
                    self.res[alg[0]] = hdb.res
                elif alg[0] == 'FOCAL':
                    foc = FOCAL_dataset(self.data, params)
                    self.res[alg[0]] = foc.res
        
        
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
        print("Dropping localisations due to photon intensity filter (%f)" % self.pc_th)
        tmp_pointcloud = pointcloud.loc[pointcloud['photon-count'] >= self.pc_th]
        pc = tmp_pointcloud[['x', 'y', 'z']]
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
        cols = ['photon-count','x', 'y', 'z']
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

    def __init__(self, data, params):

        self.alg = 'DBSCAN'
        self.min_npoints = params[3]
        self.epsilon = params[4]
        self.min_cluster_points = params[5]
        self.res = []
        for i,val in enumerate(data):
            self.call_DBSCAN(val['filename'], val['pointcloud'])
            

    def call_DBSCAN(self, filename, scanned_data):
        centroids, groups, noise = dbscan_cluster_and_group(
            xyz = scanned_data,
            min_npoints = self.min_npoints,
            eps = self.epsilon,
            min_cluster_points = self.min_cluster_points
            )
        for i, elem in enumerate(groups):
            print('group ', i, ':\n', elem)
        self.res.append({'filename': filename, 'centroids': centroids, 'groups': groups, 'noise': noise})



class HDBSCAN_dataset(dstorm_dataset):

    def __init__(self, data, params):
        
        self.alg = 'HDBSCAN'
        self.min_cluster_points = params[3]
        self.epsilon_threshold = params[4]
        self.min_samples = params[5]
        self.extracting_alg = params[6]
        self.alpha = params[7]
        self.res = []
        for i,val in enumerate(data):
            self.call_HDBSCAN(val['filename'], val['pointcloud'])


    def call_HDBSCAN(self, filename, scanned_data):
        centroids, groups, noise = hdbscan_cluster_and_group(
            xyz = self.scanned_data,
            min_cluster_points = self.min_cluster_points,
            epsilon_threshold = self.epsilon_threshold,
            min_samples = self.min_samples,
            extracting_alg = self.extracting_alg,
            alpha = self.alpha
            )
        self.res.append({'filename': filename, 'centroids': centroids, 'groups': groups, 'noise': noise})



class FOCAL_dataset(dstorm_dataset):

    def __init__(self, data, params):
        
        self.alg = 'FOCAL'
        self.sigma = params[3]
        self.minL = params[4]
        self.minC = params[5]
        self.res = []
        for i,val in enumerate(data):
            self.call_FOCAL(val['filename'], val['pointcloud'])

    def call_FOCAL(self, filename, scanned_data):
        centroids, groups, noise = focal_cluster_and_group(
            xyz = scanned_data,
            sigma = self.sigma,
            minL = self.minL,
            minC = self.minC
            )
        self.res.append({'filename': filename, 'centroids': centroids, 'groups': groups, 'noise': noise})

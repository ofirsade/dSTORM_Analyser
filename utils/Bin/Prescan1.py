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


def find_groups(df, centroids, groups, noise):
        """
        
        @param xyz: dataframe, (x,y,z) coordinates
        @param centroids: 
        @param groups: 
        @param noise: 
        ****************************************************************************************
        @ret $$$$
        """
        if len(groups) > 0:
                max_npoints = max(len(g) for g in groups)
        else:
                max_npoints = 0

        unassigned = df.iloc[noise].copy()

        
        



def call_alg(df, alg, params):
        """
        Call the chosen algorithm and run it with the specified parameters on the data in df.
        
        @param df: dataframe, containing (x, y, z) localisations and photon-count
        @param alg: str, the selected algorithm to call
        @param params: list, the parameters to input to alg
        ***********************************************
        @ret ret: list, alg output
        """
        ret = []
        pc_df = df.loc[df['photon-count'] >= params[0]] # Leave out localisations with photon-count < user defined threshold
        xyz = pc_df.to_numpy().tolist()
        print('\nXYZ[0]: ', xyz[0], '\n')
        if alg == 'DBSCAN':
                d_centroids, d_groups, d_noise = dbscan_cluster_and_group(
                        xyz = xyz,
                        min_npoints = params[3],
                        eps = params[4],
                        min_cluster_points = params[5])
                ret = [filename, d_centroids, d_groups, d_noise, 'DBSCAN']
        elif alg == 'HDBSCAN':
                h_centroids, h_groups, h_noise = hdbscan_cluster_and_group(
                        xyz = xyz,
                        min_cluster_points = params[3],
                        epsilon_threshold = params[4],
                        min_samples = params[5],
                        extracting_alg = params[6],
                        alpha = params[7])
                ret = [filename, h_centroids, h_groups, h_noise, 'HDBSCAN']
        elif alg == 'FOCAL':
                f_centroids, f_groups, f_noise = focal_cluster_and_group(
                        xyz = xyz,
                        sigma = params[3],
                        minL = params[4],
                        minC = params[5])
                ret = [filename, f_centroids, f_groups, f_noise, 'FOCAL']
        tbd = find_groups(pc_df, ret[1], ret[2], ret[3])
        
        return ret


def prep_data(path, algs, config):
        """
        Get all relevant data from file(s) input by the user and run the chosen algorithms with the specified parameters.
        
        @param path: str, path to file(s) or directory
        @param algs: str, the algorithms that the user has decided to run on the files in path
        @param config: dict, the parameters for each algorithm that is to be used according to algs
        ---------------------------------------------------------------
        @ret res: list of lists, the results of the cluster identification algorithms
        """
        cols = ['photon-count','x', 'y', 'z']
        res = []
        print('CONFIG:\n', config)
        if path[0] == 'file':
                head, filename = os.path.split((path[1])[0])
                df = pd.read_csv((path[1])[0], usecols = cols)
                for alg in algs:
                        print('ALGORITHM: ', alg, " ", type(alg))
                        if alg[1] == True:
                                params = config.get(str(alg[0])) # Get the parameters the user set for this algorithm
                                print('Params:\n', params)
                                ret = call_alg(df, alg[0], params)
                                res.append(ret)

        elif path[0] == 'files':
                for fp in (path[1])[0]:
                        df = pd.read_csv(fp, usecols = cols)
                        head, filename = os.path.split(fp)
                        for alg in algs:
                                if alg[1] == True:
                                        params = config.get(str(alg[0])) # Get the parameters the user set for this algorithm
                                        ret = call_alg(df, alg[0], params)
                                        res.append(ret)


        elif path[0] == 'dir':
                for filename in os.listdir((path[1])[0]):
                        if filename.endswith(".csv"):
                                fp = os.path.join(path[1], filename)
                                df = pd.read_csv(fp, usecols = cols)
                                for alg in algs:
                                        if alg[1] == True:
                                                params = config.get(str(alg)) # Get the parameters the user set for this algorithm
                                                ret = call_alg(df, alg[0], params)
                                                res.append(ret)

        
        return res

class dstorm_dataset(data.Dataset):

    def __init__(self, path, workers = 8):

        self.path = path[1]
        fileslist = []
        if path[0] == 'file':
            fileslist.append(self.path)
        elif path[0] == 'files':
            for full_path in self.path[0]:
                fileslist.append(full_path)
        elif path[0] == 'dir':
            for filename in os.listdir(self.path[0]):
                if filename.endswith(".csv"):
                    full_path = os.path.join(self.path, filename)
                    fileslist.append(full_path)
        else:
            raise ValueError(f"{self.path input path is not a path to a directory or a 'csv' file")

        pool = Pool(workers)
        try:
            df_rows = []
            with tqdm(total=len(fileslist)) as pbar:
                result = pool.imap_unordered(self.parse_row, fileslist)
                for r in result:
                    df_rows.append(r)
                    pbar.update(1)
        finally:
            pool.close()
            pool.terminate()

        self.orig_df = pd.DataFrame(df_rows)

        if self.orig_df is not None:
            self.data = self.orig_df.to_dict(orient = 'records')
            self.indexes = self.orig_df.index.values
            self.index_to_row = {i: dpoint for (i, dpoint) in zip(self.indexes, self.data)}
        else:
            self.data = None
            self.indexes = None
            self.index_to_row = None

    def __len__(self):
        return len(self.indexes)

    def create_groups_df(self, df):
        return None

    def process_pointcloud_df(self, pc):
        """
        Manipulate pointcloud DataFrame
        
        @param pc: dataframe, read from input file.csv
        **********************************************
        @ret df_row: dict, with one dataframe value and one int value
        """
        df_row = {'pointcloud': pc, 'num_of_points': len(pc)}
        return df_row

    
    def parse_full_path(self, full_path):
        """
        Change to parse additional/other fields from full path
        """
        parsed_full_path = {'filename': os.path.basename(full_path)}
        return parsed_full_path

    
    def parse_row(self, full_path):
        """
        Row parser -
            args (list): list of all arguments the function is getting.
                            passing arguments this way is necessary for pool.imap_unordered
        @param full_path: str, path to input file, file name, and pointcloud
        ***********************************************************************************
        @ret row: dict, containing path, 
        """
        row = {'full_path': full_path}
        cols = ['photon-count','x', 'y', 'z']
        try:
            parsed_full_path = self.parse_full_path(row['full_path'])

            with open(row['full_path'], 'r') as f:
                processed_pointcloud = self.process_pointcloud_df(read_csv(f, usecols = cols))

            row = {**row, **parsed_full_path, **processed_pointcloud}

        except Exception as e:
            row['Exception'] = format_exc()

        return row


class dstorm_subset(dstorm_dataset):

    def __init__(self,
                 *args,
                 coloc_distance = 50,
                 coloc_neighbors = 1,
                 use_z = False,
                 noise_reduce = False,
                 stddev_num = 1.0,
                 density_drop_threshold = 0.0,
                 z_density_drop_threshold = 0.0,
                 photon_count = 0.0,
                 **kwargs):

        self.coordinates_vector = ['x', 'y', 'z'] if use_z else ['x', 'y']
        self.max_npoints = 0

        #print(self.coordinates_vector)
        
        # Colocalization
        self.coloc_distance = coloc_distance
        self.coloc_neighbors = coloc_neighbors
        self.use_z = use_z
        self.noise_reduce = noise_reduce
        self.stddev_num = stddev_num
        self.density_drop_threshold = density_drop_threshold
        self.z_density_drop_threshold = z_density_drop_threshold
        self.photon_count = photon_count
        kwargs.pop("use_z") # This is done to respect the super's interface

        super(dstorm_subset, self).__init__(*args, **kwargs)
                                
    def parse_full_path(self, full_path):
        parsed_full_path = super(_ColocDstormDataset, self).parse_full_path(full_path)


    def create_groups_df(self, df):
      


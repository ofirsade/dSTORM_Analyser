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
        print("Dropping due to photon intensity filter (%f)" % params[0])
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

import numpy as np
import pandas as pd
import os
import torch.utils.data as data

from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from traceback import format_exc
from pandas.io.parsers import read_csv
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QProgressBar, QVBoxLayout, QApplication
from pathlib import Path

from utils.Extract import extract_AP, pca_outliers_and_axes
from utils.Cluster_Identification_Algorithms import sample_and_group, dbscan_cluster_and_group,\
     hdbscan_cluster_and_group, focal_cluster_and_group
from utils.Plot import plot_res
import time
import sys



class dstorm_dataset(data.Dataset):
    
    def __init__(self, input_path, csvs_path, htmls_path, selected_files, algs, config, open_plots, pbar, workers = 8):
        '''
        @params:
            input_path - str, path to file/files/directory
            output_path - str, path to desired output directory
            algs - list of bools, [DBSCAN, HDBSCAN, FOCAL]
            config - list, configuration parameters for each algorithm set to TRUE in algs
            open_plots - bool, if True: open plots when ready, elif False: only save the plots to output directory
            pw - PlotWindow, empty window to show results
            pbar - progress bar
        '''
        fileslist = []
        self.input_path = input_path[1][0]
        self.pc_th = 0.0
        self.px_th = 1000.0
        self.clust_res = {}
        self.error = None
        self.complete = False

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
##                    pb = PBar()
##                    pb.show()
                    
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
                    db = DBSCAN_dataset(self.data, params, csvs_path, htmls_path, open_plots)
                    self.clust_res[alg[0]] = db.res
                elif alg[0] == 'HDBSCAN':
                    hdb = HDBSCAN_dataset(self.data, params, csvs_path, htmls_path, open_plots)
                    self.clust_res[alg[0]] = hdb.res
                elif alg[0] == 'FOCAL':
                    foc = FOCAL_dataset(self.data, params, csvs_path, htmls_path, open_plots)
                    self.clust_res[alg[0]] = foc.res
##                pbar.setValue(i)
        self.complete = True

        
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
        self.orig_loc_num = len(pointcloud.index) # Original number of points before any filteration
        
        tmp_pointcloud = pointcloud.loc[pointcloud['photon-count'] >= self.pc_th]
        pc_dp = self.orig_loc_num - len(tmp_pointcloud) # Number of points dropped by photoncount filter
        print("Dropping " + str(pc_dp) + " localisations due to photon intensity filter (%f)" % self.pc_th)
        
        fin_pointcloud = tmp_pointcloud.loc[tmp_pointcloud['precisionx'] <= self.px_th]
        xp_dp = len(tmp_pointcloud) - len(fin_pointcloud) # Number of points dropped by x-precision filter
        print("Dropping " + str(xp_dp) + " localisations due to x-precision filter (%f)" % self.px_th)

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

    def __init__(self, data, params, csvs_path, htmls_path, show_plts):
        '''
        Creates an object of 3D localisations, each with a cluster label given by DBSCAN
        @param data: 
        @param params:
        '''

        self.alg = 'DBSCAN'
        self.epsilon = params[4]
        self.min_samples = params[5]
        self.csvs_path = csvs_path
        self.htmls_path = htmls_path
        self.show_plts = show_plts
        if len(params) <= 6: # if the user didn't set a PCA standard deviation
            self.pca_stddev = None
        else: # if the user did set a PCA standard deviation
            self.pca_stddev = params[6]
        self.res = {}
        for i,val in enumerate(data):            
            self.call_DBSCAN(val['filename'], val['pointcloud'])
        


    def call_DBSCAN(self, filename, scanned_data):
        """
        Calls DBSCAN and creates self.res - the resulting array of localisations with cluster labels
        @param filename: the filename of the data to run HDBSCAN on
        @param scanned_data: the localisations to run HDBSCAN on
        """

        print('Filename: ', filename)
        data_df = scanned_data.copy()
        tmp_df = data_df[['x', 'y', 'z']]
        if self.pca_stddev == None: # If the user didn't select denoise with PCA
            xyz_df = tmp_df
        else:
            tmp_arr = tmp_df.to_numpy()
            denoised_xyz_arr = pca_outliers_and_axes(tmp_arr, self.pca_stddev) # Denoise with PCA
            xyz_df = pd.DataFrame(denoised_xyz_arr, columns = ['x', 'y', 'z'])
            
        img_props, cluster_props, cluster_props_dict, xyzl = dbscan_cluster_and_group(xyz = xyz_df,
                                                                                      eps = self.epsilon,
                                                                                      min_samples = self.min_samples,
                                                                                      fname = filename)
        if len(img_props.index) > 0:
            
            new_fn = 'DBSCAN_' + str(self.min_samples) + '_' + str(self.epsilon)

            fname = str(Path(filename).stem)

            now = datetime.now() # datetime object containing current date and time
            dt_string = now.strftime("%Y.%m.%d %H_%M_%S")
            
            img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
            cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'
            img_path = os.path.join(self.csvs_path, '', img_props_name)
            cluster_path = os.path.join(self.csvs_path, '', cluster_props_name)

##            img_props.insert(loc = 1,
##                             column = 'Number of Localisations Filtered by X-Precision',
##                             value = [xp_dp])
##            img_props.insert(loc = 1,
##                             column = 'Number of Localisations Filtered by Photoncount',
##                             value = [pc_dp])
##            img_props.insert(loc = 1,
##                             column = 'Original Number of Localisations',
##                             value = [orig_loc_num])

            img_props.to_excel(img_path)
            cluster_props.to_excel(cluster_path)
            plot_res(xyzl, cluster_props_dict, fname, 'DBSCAN', self.htmls_path, dt_string, self.show_plts)
            
        else:
            print('No Clusters were found in DBSCAN!')
        


class HDBSCAN_dataset(dstorm_dataset):

    def __init__(self, data, params, csvs_path, htmls_path, show_plts):
        
        self.alg = 'HDBSCAN'
        self.min_cluster_points = params[4]
        self.epsilon_threshold = params[5]
        self.min_samples = params[6]
        self.extracting_alg = params[7]
        self.alpha = params[8]
##        self.output_path = output_path
        self.csvs_path = csvs_path
        self.htmls_path = htmls_path 
        self.show_plts = show_plts

        if len(params) <= 9: # if the user didn't set a PCA standard deviation
            self.pca_stddev = None
        else: # if the user did set a PCA standard deviation
            self.pca_stddev = params[9]
        self.res = []
        for i,val in enumerate(data):
            self.call_HDBSCAN(val['filename'], val['pointcloud'])



    def call_HDBSCAN(self, filename, scanned_data):
        """
        Calls HDBSCAN and creates self.res - the resulting array of localisations with cluster labels
        @param filename: the filename of the data to run HDBSCAN on
        @param scanned_data: the localisations to run HDBSCAN on
        """

        data_df = scanned_data.copy()
        tmp_df = data_df[['x', 'y', 'z']]
        if self.pca_stddev == None: # If the user didn't select denoise with PCA
            xyz_df = tmp_df
        else:
            tmp_arr = tmp_df.to_numpy()
            denoised_xyz_arr = pca_outliers_and_axes(tmp_arr, self.pca_stddev) # Denoise with PCA
            xyz_df = pd.DataFrame(denoised_xyz_arr, columns = ['x', 'y', 'z'])
        
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
            
            now = datetime.now() # datetime object containing current date and time
            dt_string = now.strftime("%Y.%m.%d %H_%M_%S")

            img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
            cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'
            img_path = os.path.join(self.csvs_path, img_props_name)
            cluster_path = os.path.join(self.csvs_path, cluster_props_name)

##            img_props.insert(loc = 1,
##                             column = 'Number of Localisations Filtered by X-Precision',
##                             value = [self.xp_dp])
##            img_props.insert(loc = 1,
##                             column = 'Number of Localisations Filtered by Photoncount',
##                             value = [self.pc_dp])
##            img_props.insert(loc = 1,
##                             column = 'Original Number of Localisations',
##                             value = [self.orig_loc_num])
            
##            img_props.to_excel(self.output_path + '/' + img_props_name)
##            cluster_props.to_excel(self.output_path + '/' + cluster_props_name)
            img_props.to_excel(img_path)
            cluster_props.to_excel(cluster_path)
            
            plot_res(xyzl, cluster_props_dict, fname, 'HDBSCAN', self.htmls_path, dt_string, self.show_plts)
            
        else:
            print('No clusters were found in HDBSCAN!')
        



class FOCAL_dataset(dstorm_dataset):

    def __init__(self, data, params, csvs_path, htmls_path, show_plts):
        
        self.alg = 'FOCAL'
        self.sigma = params[4]
        self.minL = params[5]
        self.minC = params[6]
        self.minPC = params[7]
        self.show_plts = show_plts
##        self.output_path = output_path
        self.csvs_path = csvs_path
        self.htmls_path = htmls_path
        if len(params) <= 8: # if the user didn't set a PCA standard deviation
            self.pca_stddev = None
        else: # if the user did set a PCA standard deviation
            self.pca_stddev = params[8]
        self.res = []
        for i,val in enumerate(data):
            self.call_FOCAL(val['filename'], val['pointcloud'])

    def call_FOCAL(self, filename, scanned_data):

        data_df = scanned_data.copy()
        tmp_df = data_df[['x', 'y', 'z']]
        if self.pca_stddev == None: # If the user didn't select denoise with PCA
            xyz_df = tmp_df
        else:
            tmp_arr = tmp_df.to_numpy()
            denoised_xyz_arr = pca_outliers_and_axes(tmp_arr, self.pca_stddev) # Denoise with PCA
            denoised_df = pd.DataFrame(denoised_xyz_arr, columns = ['x', 'y', 'z'])
            xyz_df = pd.merge(denoised_df, data_df, on = ['x', 'y', 'z'], how = 'inner')
        
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
            dt_string = now.strftime("%Y.%m.%d %H_%M_%S") # YY/mm/dd H:M:S
            
            img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
            cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'
            img_path = os.path.join(self.csvs_path, img_props_name)
            cluster_path = os.path.join(self.csvs_path, cluster_props_name)

##            img_props.insert(loc = 1,
##                             column = 'Number of Localisations Filtered by X-Precision',
##                             value = [self.xp_dp])
##            img_props.insert(loc = 1,
##                             column = 'Number of Localisations Filtered by Photoncount',
##                             value = [self.pc_dp])
##            img_props.insert(loc = 1,
##                             column = 'Original Number of Localisations',
##                             value = [self.orig_loc_num])
        
##            img_props.to_excel(self.output_path + '/' + img_props_name)
##            cluster_props.to_excel(self.output_path + '/' + cluster_props_name)
            img_props.to_excel(img_path)
            cluster_props.to_excel(cluster_path)
            
            plot_res(xyzl, cluster_props_dict, fname, 'FOCAL', self.htmls_path, dt_string, self.show_plts)

        else:
            print('No clusters were found in FOCAL!')


class Thread(QThread):
    _signal = pyqtSignal(int)
    def __init__(self):
        super(Thread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        for i in range(100):
            time.sleep(0.2)
            self._signal.emit(i)

class PBar(QWidget):
    def __init__(self):
        super(PBar, self).__init__()
        self.setWindowTitle('QProgressBar')
        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        self.resize(300, 100)
        self.vbox = QVBoxLayout(self)
        self.vbox.addWidget(self.pbar)
        self.setLayout(self.vbox)
        self.pbFunc()
        self.show()

    def pbFunc(self):
        self.thread = Thread()
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()

    def signal_accept(self, msg):
        self.pbar.setValue(int(msg))
        if self.pbar.value() == 99:
            self.pbar.setValue(0)



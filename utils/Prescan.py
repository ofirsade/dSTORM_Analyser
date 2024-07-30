import numpy as np
import pandas as pd
import os
import torch.utils.data as data

from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from traceback import format_exc
from pandas.io.parsers import read_csv
from datetime import datetime
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QPushButton, QProgressBar, QVBoxLayout, QApplication
from pathlib import Path

from utils.Extract import extract_AP, pca_outliers_and_axes
from utils.Cluster_Identification_Algorithms import sample_and_group, dbscan_cluster_and_group,\
     hdbscan_cluster_and_group, focal_cluster_and_group
from utils.Plot import plot_res, plot_2D_image
import time
import sys


##class ThreadClass(QThread):
##    progress = pyqtSignal(object)
##
##    def __init__(self, parent = None):
##        super(MainWindow, self).__init__(parent)
##
##    def run(self):
##        self.ds = dstorm_dataset()
##        
    
class dstorm_dataset(data.Dataset):
    
    def __init__(self, input_path, csvs_path, htmls_path, selected_files, algs, config,
                 gen_files, gen_plots, open_plots, workers = 8):# pbar, workers = 8):
        '''
        @params:
            input_path - str, path to file/files/directory
            output_path - str, path to desired output directory
            algs - list of bools, [DBSCAN, HDBSCAN, FOCAL]
            config - list, configuration parameters for each algorithm set to TRUE in algs
            gen_files - bool, if True: generate and save spreadsheets with clustering information, else don't
            gen_plots - bool, if True: generate and save htmls of clustering plots, else don't
            open_plots - bool, if True: open plots when ready, elif False: only save the plots to output directory
            pw - PlotWindow, empty window to show results
##            pbar - progress bar
        '''
##        self.popup = PopUpProgressB()
        fileslist = []
        self.input_path = input_path[1]
        self.pc_th = 0.0
        self.px_th = 1000.0
        self.clust_res = {}
        self.error = None
        self.complete = False
##        l = 0

        for alg in algs:
            if alg[1] == True:
                self.pc_th = (config.get(str(alg[0])))[0] # Set the minimum photon-count threshold
                self.px_th = (config.get(str(alg[0])))[1] # Set the maximum x-precision threshold
##                l += 1

        if input_path[0] == 'files':
            self.input_path = self.input_path[0]
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

##        fl = len(fileslist) * l * 100
##        print('Number of Analyses: ', fl)

        pool = Pool(workers)
        
        try:
            df_rows = []
            with tqdm(total = len(fileslist)) as progbar:
                result = pool.imap_unordered(self.parse_row, fileslist) # Load files
                for r in result:
                    df_rows.append(r)
                    
        finally:
            pool.close()
            pool.terminate()

        self.orig_df = pd.DataFrame(df_rows)
        j = 0
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
                    db = DBSCAN_dataset(self.data, params, csvs_path, htmls_path, gen_files, gen_plots, open_plots)
                    self.clust_res[alg[0]] = [db.img_props, db.cluster_props]
                elif alg[0] == 'HDBSCAN':
                    hdb = HDBSCAN_dataset(self.data, params, csvs_path, htmls_path, gen_files, gen_plots, open_plots)
                    self.clust_res[alg[0]] = [hdb.img_props, hdb.cluster_props]
                elif alg[0] == 'FOCAL':
                    foc = FOCAL_dataset(self.data, params, csvs_path, htmls_path, gen_files, gen_plots, open_plots)
                    self.clust_res[alg[0]] = [foc.img_props, foc.cluster_props]

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
        orig_loc_num = len(pointcloud.index) # Original number of points before any filteration
        print('\nInitial number of localisations: ', orig_loc_num, '\n')
        
        tmp_pointcloud = pointcloud.loc[pointcloud['photon-count'] >= self.pc_th]
        pc_dp = orig_loc_num - len(tmp_pointcloud) # Number of points dropped by photoncount filter
        print("\nDropping " + str(pc_dp) + " localisations due to photon intensity filter (%f)" % self.pc_th + '\n')
        
        fin_pointcloud = tmp_pointcloud.loc[tmp_pointcloud['precisionx'] <= self.px_th]
        xp_dp = len(tmp_pointcloud) - len(fin_pointcloud) # Number of points dropped by x-precision filter
        print("\nDropping " + str(xp_dp) + " localisations due to x-precision filter (%f)" % self.px_th + '\n')

        pc = fin_pointcloud[['x', 'y', 'z', 'photon-count']]
        
        df_row = {'pointcloud': pc, 'num_of_points': orig_loc_num}
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

    def __init__(self, data, params, csvs_path, htmls_path, gen_files, gen_plots, show_plts):
        '''
        Creates an object of 3D localisations, each with a cluster label given by DBSCAN
        @param data: 
        @param params:
        '''

        self.alg = 'DBSCAN'
        self.d2_th = params[2]
        self.d3_th = params[3]
        self.epsilon = params[4]
        self.min_samples = params[5]
        self.csvs_path = csvs_path
        self.htmls_path = htmls_path
        self.gen_files = gen_files
        self.gen_plts = gen_plots
        self.show_plts = show_plts
        self.img_props = pd.DataFrame()
        self.cluster_props = pd.DataFrame()
        if len(params) <= 6: # if the user didn't set a PCA standard deviation
            self.pca_stddev = None
        else: # if the user did set a PCA standard deviation
            self.pca_stddev = params[6]
        self.res = {}
        for i,val in enumerate(data):
##            tmp_df = val['pointcloud']
##            tmp_df.sort_values(by=['x', 'y'], inplace = True)
##            print('\n', val['pointcloud'], '\n')
            self.call_DBSCAN(val['filename'], val['pointcloud'], val['num_of_points'])
        


    def call_DBSCAN(self, filename, scanned_data, ln):
        """
        Calls DBSCAN and creates self.res - the resulting array of localisations with cluster labels
        @param filename: the filename of the data to run HDBSCAN on
        @param scanned_data: the localisations to run HDBSCAN on
        @param ln: int, the number of localisations in the original dataframe
        """

        print('Filename: ', filename)
        data_df = scanned_data.copy()
##        print('\nData DF:\n', data_df)
        tmp_df = data_df[['x', 'y', 'z']]
##        if self.pca_stddev == None: # If the user didn't select denoise with PCA
##            xyz_df = tmp_df
##        else:
##            tmp_arr = tmp_df.to_numpy()
##            denoised_xyz_arr = pca_outliers_and_axes(tmp_arr, self.pca_stddev) # Denoise with PCA
##            xyz_df = pd.DataFrame(denoised_xyz_arr, columns = ['x', 'y', 'z'])
        xyz_df = tmp_df
##        print('\nXYZ DF:\n', xyz_df)
        img_props, cluster_props, cluster_props_dict, xyzl = dbscan_cluster_and_group(xyz = xyz_df,
                                                                                      eps = self.epsilon,
                                                                                      min_samples = self.min_samples,
                                                                                      fname = filename,
                                                                                      pca_stddev = self.pca_stddev,
                                                                                      d2_th = self.d2_th,
                                                                                      d3_th = self.d3_th)
        
##        print('\nXYZL DF:\n', xyzl)
        
        if len(img_props.index) > 0:
            
            new_fn = 'DBSCAN_' + str(self.min_samples) + '_' + str(self.epsilon)

            fname = str(Path(filename).stem)

            now = datetime.now() # datetime object containing current date and time
            dt_string = now.strftime("%Y.%m.%d %H_%M_%S")
            
##            img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
##            cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'
##            img_path = os.path.join(self.csvs_path, '', img_props_name)
##            cluster_path = os.path.join(self.csvs_path, '', cluster_props_name)

##            img_props.insert(loc = 1,
##                             column = 'Number of Localisations Filtered by X-Precision',
##                             value = [xp_dp])
##            img_props.insert(loc = 1,
##                             column = 'Number of Localisations Filtered by Photoncount',
##                             value = [pc_dp])
##            img_props.insert(loc = 1,
##                             column = 'Original Number of Localisations',
##                             value = [orig_loc_num])

            img_props.insert(loc = 2,
                             column = 'Original Number of Localisations',
                             value = [ln])
            img_props['PCA'] = [self.pca_stddev]
##            if self.gen_files:
##                img_props['PCA'] = [self.pca_stddev]
##                img_props.to_excel(img_path)
##                cluster_props.to_excel(cluster_path)

            if self.gen_plts:
                plot_res(xyzl, cluster_props_dict, fname, 'DBSCAN', self.htmls_path, dt_string, self.show_plts)
##                plot_2D_image(xyzl, fname, self.htmls_path, dt_string, self.show_plts)

            self.img_props = pd.concat([self.img_props, img_props])
            self.cluster_props = pd.concat([self.cluster_props, cluster_props])
            
        else:
            print('No Clusters were found in DBSCAN!')
        


class HDBSCAN_dataset(dstorm_dataset):

    def __init__(self, data, params, csvs_path, htmls_path, gen_files, gen_plots, show_plts):
        
        self.alg = 'HDBSCAN'
        self.d2_th = params[2]
        self.d3_th = params[3]
        self.min_cluster_points = params[4]
        self.epsilon_threshold = params[5]
        self.min_samples = params[6]
        self.extracting_alg = params[7]
        self.alpha = params[8]
        self.min_cluster_size = params[9]
##        self.output_path = output_path
        self.csvs_path = csvs_path
        self.htmls_path = htmls_path
        self.gen_files = gen_files
        self.gen_plts = gen_plots
        self.show_plts = show_plts
        self.img_props = pd.DataFrame()
        self.cluster_props = pd.DataFrame()

        if len(params) <= 9: # if the user didn't set a PCA standard deviation
            self.pca_stddev = None
        else: # if the user did set a PCA standard deviation
            self.pca_stddev = params[9]
        self.res = []
        for i,val in enumerate(data):
##            print('\n', val['pointcloud'], '\n')
            self.call_HDBSCAN(val['filename'], val['pointcloud'], val['num_of_points'])



    def call_HDBSCAN(self, filename, scanned_data, ln):
        """
        Calls HDBSCAN and creates self.res - the resulting array of localisations with cluster labels
        @param filename: the filename of the data to run HDBSCAN on
        @param scanned_data: the localisations to run HDBSCAN on
        @param ln: int, the number of localisations in the original dataframe
        """

        data_df = scanned_data.copy()
        tmp_df = data_df[['x', 'y', 'z']]
##        if self.pca_stddev == None: # If the user didn't select denoise with PCA
##            xyz_df = tmp_df
##        else:
##            tmp_arr = tmp_df.to_numpy()
##            denoised_xyz_arr = pca_outliers_and_axes(tmp_arr, self.pca_stddev) # Denoise with PCA
##            xyz_df = pd.DataFrame(denoised_xyz_arr, columns = ['x', 'y', 'z'])
        xyz_df = tmp_df
        
        img_props, cluster_props, cluster_props_dict, xyzl = hdbscan_cluster_and_group(
            xyz = xyz_df,
            min_cluster_points = self.min_cluster_points,
            epsilon_threshold = self.epsilon_threshold,
            min_samples = self.min_samples,
            extracting_alg = self.extracting_alg,
            alpha = self.alpha,
            fname = filename,
            pca_stddev = self.pca_stddev,
            d2_th = self.d2_th,
            d3_th = self.d3_th,
            min_cluster_size = self.min_cluster_size
            )

        if len(img_props.index) > 0:
            new_fn = 'HDBSCAN_' + str(self.epsilon_threshold)
            fname = str(Path(filename).stem)
            
            now = datetime.now() # datetime object containing current date and time
            dt_string = now.strftime("%Y.%m.%d %H_%M_%S")

##            img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
##            cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'
##            img_path = os.path.join(self.csvs_path, img_props_name)
##            cluster_path = os.path.join(self.csvs_path, cluster_props_name)

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

            img_props.insert(loc = 2,
                             column = 'Original Number of Localisations',
                             value = [ln])
            img_props['PCA'] = [self.pca_stddev]
##            if self.gen_files:
##                img_props['PCA'] = [self.pca_stddev]
##                img_props.to_excel(img_path)
##                cluster_props.to_excel(cluster_path)

            self.img_props = pd.concat([self.img_props, img_props])
            self.cluster_props = pd.concat([self.cluster_props, cluster_props])

            if self.gen_plts:
                plot_res(xyzl, cluster_props_dict, fname, 'HDBSCAN', self.htmls_path, dt_string, self.show_plts)
##                plot_2D_image(xyzl, fname, self.htmls_path, dt_string, self.show_plts)
            
        else:
            print('No clusters were found in HDBSCAN!')
        



class FOCAL_dataset(dstorm_dataset):

    def __init__(self, data, params, csvs_path, htmls_path, gen_files, gen_plots, show_plts):
        
        self.alg = 'FOCAL'
        self.d2_th = params[2]
        self.d3_th = params[3]
        self.sigma = params[4]
        self.minL = params[5]
        self.minC = params[6]
        self.minPC = params[7]
        self.gen_files = gen_files
        self.gen_plts = gen_plots
        self.show_plts = show_plts
##        self.output_path = output_path
        self.csvs_path = csvs_path
        self.htmls_path = htmls_path
        self.img_props = pd.DataFrame()
        self.cluster_props = pd.DataFrame()

        if len(params) <= 8: # if the user didn't set a PCA standard deviation
            self.pca_stddev = None
        else: # if the user did set a PCA standard deviation
            self.pca_stddev = params[8]
        self.res = []
        for i,val in enumerate(data):
##            print('\n', val['pointcloud'], '\n')
            self.call_FOCAL(val['filename'], val['pointcloud'], val['num_of_points'])

    def call_FOCAL(self, filename, scanned_data, ln):

        """
        Calls FOCAL and creates self.res - the resulting array of localisations with cluster labels
        @param filename: the filename of the data to run FOCAL on
        @param scanned_data: the localisations to run FOCAL on
        @param ln: int, the number of localisations in the original dataframe
        """

        data_df = scanned_data.copy()
        tmp_df = data_df[['x', 'y', 'z']]
##        if self.pca_stddev == None: # If the user didn't select denoise with PCA
####            xyz_df = tmp_df
##            xyz_df = data_df[['x', 'y', 'z', 'photon-count']]
##        else:
##            tmp_arr = tmp_df.to_numpy()
##            denoised_xyz_arr = pca_outliers_and_axes(tmp_arr, self.pca_stddev) # Denoise with PCA
##            denoised_df = pd.DataFrame(denoised_xyz_arr, columns = ['x', 'y', 'z'])
##            xyz_df = pd.merge(denoised_df, data_df, on = ['x', 'y', 'z'], how = 'inner')
        xyz_df = data_df[['x', 'y', 'z', 'photon-count']]
        
        img_props, cluster_props, cluster_props_dict, xyzl = focal_cluster_and_group(
            xyz = xyz_df,
            sigma = self.sigma,
            minL = self.minL,
            minC = self.minC,
            minPC = self.minPC,
            fname = filename,
            pca_stddev = self.pca_stddev,
            d2_th = self.d2_th,
            d3_th = self.d3_th
            )

        if len(img_props.index) > 0:
            new_fn = 'FOCAL_' + str(self.sigma) + '_' + str(self.minL) + '_' + str(self.minC)
            fname = str(Path(filename).stem)
            
            now = datetime.now() # datetime object containing current date and time
            dt_string = now.strftime("%Y.%m.%d %H_%M_%S") # YY/mm/dd H:M:S
##            
##            self.img_props_name = dt_string + ' ' + fname + ' Image ' + new_fn + '.xlsx'
##            self.cluster_props_name = dt_string + ' ' + fname + ' Cluster ' + new_fn + '.xlsx'
##            
##            img_path = os.path.join(self.csvs_path, img_props_name)
##            cluster_path = os.path.join(self.csvs_path, cluster_props_name)

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
            img_props.insert(loc = 2,
                             column = 'Original Number of Localisations',
                             value = [ln])
            img_props['PCA'] = [self.pca_stddev]
##            if self.gen_files:
##                img_props['PCA'] = [self.pca_stddev]
##                img_props.to_excel(img_path)
##                cluster_props.to_excel(cluster_path)

            self.img_props = pd.concat([self.img_props, img_props])
            self.cluster_props = pd.concat([self.cluster_props, cluster_props])
            
            if self.gen_plts:
                plot_res(xyzl, cluster_props_dict, fname, 'FOCAL', self.htmls_path, dt_string, self.show_plts)
##                plot_2D_image(xyzl, fname, self.htmls_path, dt_string, self.show_plts)

        else:
            print('No clusters were found in FOCAL!')


class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int)

    @pyqtSlot()
    def proc_counter(self):  # A slot takes no params
        for i in range(1, 100):
            time.sleep(0.1)
            self.intReady.emit(i)

        self.finished.emit()


##class PopUpProgressB(QWidget):
##
##    def __init__(self):
##        super().__init__()
##        self.pbar = QProgressBar(self)
##        self.pbar.setGeometry(30, 40, 500, 75)
##        self.layout = QVBoxLayout()
##        self.layout.addWidget(self.pbar)
##        self.setLayout(self.layout)
##        self.setGeometry(300, 300, 550, 100)
##        self.setWindowTitle('Progress Bar')
####        self.statusBar().addPermanentWidget(self.pbar)
##        self.show()
##
##        self.running()
##
####        self.obj = Worker()
####        self.thread = QThread()
####        self.obj.intReady.connect(self.on_count_changed)
####        self.obj.moveToThread(self.thread)
####        self.obj.finished.connect(self.thread.quit)
####        self.thread.started.connect(self.obj.proc_counter)
####        self.thread.start()
##
##    def on_count_changed(self, value):
##        self.pbar.setValue(value)
##
##    def running(self):
##        self.completed = 0
####        self.statusBar().showMessage("downloading ...", 0)
## 
##        while self.completed < 100:
##            self.completed += 0.005
##            self.pbar.setValue(int(self.completed))
##             
##            if self.pbar.value() == 100:
####                self.statusBar().showMessage("completed", 0)
##                self.pbar.hide()



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

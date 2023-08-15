
from PyQt5.QtWidgets import *
"""QApplication, QMainWindow, QLabel, QGridLayout, \
  QWidget, QPushButton, QTableWidget, QTableWidgetItem, QInputDialog, \
  QMessageBox, QComboBox, QLineEdit, QAction, QCheckBox, QFileDialog, QVBoxLayout"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import * #QIcon
from PyQt5.QtCore import * #pyqtSlot, QThread, pyqtSignal
import pyqtgraph as pg
import sys, os
from os import listdir, path, walk, mkdir, unlink, environ
from shutil import rmtree, copyfile
from datetime import datetime
import time
import re
from utils.Prescan import dstorm_dataset
import multiprocessing

multiprocessing.set_start_method('forkserver', force = True)
multiprocessing.freeze_support()



class AlignDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter

class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)

        
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.init_UI()
        self.algs = [['DBSCAN',False], ['HDBSCAN', False], ['FOCAL', False]] # Which algorithm(s) were selected by the user
        self.config = dict()
        self.prep_data = 0
    
    def init_UI(self):

        self.path = []
        self.checked_cbs = []
        self.open_plots = False
##        self.output_dir = os.path.abspath(os.path.dirname(__file__))
        self.output_dir_set = False
        scriptDir = os.path.dirname(os.path.realpath(__file__))
##        self.basedir = os.path.dirname(__file__)
        
        self.resize(1000, 800)
##        self.setWindowIcon(QtGui.QIcon('/Users/ofirsade/Desktop/UNI/Masters/Code/Analysis_Platform/dstormlogo2.png'))
        self.setWindowTitle('dSTORM Analyser')
##        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep +  'dSTORMlogo1.png'))
##        print(scriptDir + os.path.sep +  'dstormlogo2.ico')

        # creating label
        self.label = QLabel(self)
        # loading image
        self.pixmap = QPixmap(os.path.join(scriptDir, 'icons', 'dSTORMlogo1.png'))
        self.smaller_pixmap = self.pixmap.scaled(300, 600, Qt.KeepAspectRatio, Qt.FastTransformation)
 
        # adding image to label
        self.label.setPixmap(self.smaller_pixmap)

 
        # Optional, resize label to image size
        self.label.resize(self.pixmap.width(),
                          self.pixmap.height())
        self.label.setAlignment(Qt.AlignCenter)
        
        self.main_layout = QGridLayout()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.overall_layout = QGridLayout(self.central_widget)
        self.overall_layout.addWidget(self.label, 0, 0, 1, 6)

        # changing the background color to black
##        self.setStyleSheet("background-color: black;")
##        self.setWindowFlag(Qt.FramelessWindowHint)
##        self.setAttribute(Qt.WA_TranslucentBackground)
##        self.setWindowOpacity(0.95)
##        radius = 30
##        self.central_widget.setStyleSheet(
##            """
##            background:rgb(255, 255, 255);
##            border-top-left-radius:{0}px;
##            border-bottom-left-radius:{0}px;
##            border-top-right-radius:{0}px;
##            border-bottom-right-radius:{0}px;
##            """.format(radius)
##        )
        
##        self.label = QLabel(self)#, size_hint_y = None, height = 50)
##        self.label.setText("Welcome to dSTORM Analyser")
##        my_font = QtGui.QFont('Arial', 30)
##        my_font.setBold(True)
##        self.label.setFont(QtGui.QFont(my_font))
####        self.label.adjustSize()
##        self.label.setAlignment(Qt.AlignCenter)
##
##        self.overall_layout.addWidget(Color('white'), 0, 0, 1, 6)
##
##        # start at row 0, column 0, use 1 row and 6 columns. 
##        self.overall_layout.addWidget(self.label, 0, 0, 1, 6)
####        self.overall_layout.resizeColumnToContents(0)
##
##        palette = QtGui.QPalette()
##        palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
##        self.setPalette(palette)
##        font = QtGui.QFont('Arial', 15, QtGui.QFont.Bold)
##        self.setFont(font)

##        self.gf = False ## What is self.gf?
        
        self.btn1 = QPushButton(self)
        self.btn1.setText('Upload Input Files')
        self.btn1.adjustSize()
        self.btn1.clicked.connect(self.get_files)
        self.overall_layout.addWidget(self.btn1, 1, 0)

        self.btn2 = QPushButton(self)
        self.btn2.setText('Select Input Directory')
        self.btn2.adjustSize()
        self.btn2.clicked.connect(self.get_dir)
        self.overall_layout.addWidget(self.btn2, 1, 1)

        self.show_plots_cb = QCheckBox('Show Plots')
        self.show_plots_cb.setChecked(False)
        self.show_plots_cb.stateChanged.connect(lambda:self.checked_show_plots())
        self.overall_layout.addWidget(self.show_plots_cb, 1, 2)

        self.db = QCheckBox('DBSCAN')
        self.db.setChecked(False)
        self.db.setCheckable(False)
        self.db.stateChanged.connect(lambda:self.btnstate(self.db))
        self.overall_layout.addWidget(self.db, 2, 0)

        self.hb = QCheckBox('HDBSCAN')
        self.hb.setChecked(False)
        self.hb.setCheckable(False)
        self.hb.stateChanged.connect(lambda:self.btnstate(self.hb))
        self.overall_layout.addWidget(self.hb, 2, 1)

        self.fb = QCheckBox('FOCAL')
        self.fb.setChecked(False)
        self.fb.setCheckable(False)
        self.fb.stateChanged.connect(lambda:self.btnstate(self.fb))
        self.overall_layout.addWidget(self.fb, 2, 2)

##        self.cb1 = QCheckBox('Use PCA')
##        self.cb1.setChecked(False)
####        self.cb1.stateChanged.connect(lambda:self.use_PCA(self.cb1))
##        self.cb1.stateChanged.connect(self.onStateChanged)
##        self.overall_layout.addWidget(self.cb1, 1, 2)

        self.btn6 = QPushButton(self)
        self.btn6.setText('Select Output Directory')
        self.btn6.adjustSize()
        self.btn6.clicked.connect(self.get_output_dir)
        self.btn6.setToolTip('This is where all your output data will be stored')
        self.overall_layout.addWidget(self.btn6, 6, 0)

##        self.output_path = QLineEdit('Insert Path to Output Directory', self)
##        self.overall_layout.addWidget(self.output_path, 4, 0)

        self.setLayout(self.overall_layout)
        self.show()


    def run_scan(self):
        """
        Get all relevant data from file(s) input by the user
        """
          
        self.get_params()
##        if self.output_dir != os.path.abspath(os.path.dirname(__file__)):
        if self.output_dir_set:
            htmls_path, csvs_path = self.set_output_paths(True)
            dataset = dstorm_dataset(self.path, csvs_path, htmls_path, self.selected_files, self.algs, self.config, self.open_plots, self.pbar)
##            self.thread = dstorm_dataset(self.path, csvs_path, htmls_path, self.selected_files, self.algs, self.config, self.open_plots, self.pbar)
##            self.thread.progress.connect(self.update_progress())
            self.msg = QMessageBox()
            self.msg.setText("The results of your scan(s) will be saved to the selected directory")
            self.msg.setInformativeText(str(self.output_dir))
            self.msg.setStandardButtons(QMessageBox.Ok) # seperate buttons with "|"
            self.msg.setDefaultButton(QMessageBox.Ok)  # setting default button to Ok
            self.msg.show()
            

        else:
            self.output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "Results")
            htmls_path, csvs_path = self.set_output_paths(False)
            dataset = dstorm_dataset(self.path, csvs_path, htmls_path, self.selected_files, self.algs, self.config, self.open_plots, self.pbar)
##            self.thread = dstorm_dataset(self.path, csvs_path, htmls_path, self.selected_files, self.algs, self.config, self.open_plots, self.pbar)
##            self.thread.progress.connect(self.update_progress())
            self.msg = QMessageBox()
            self.msg.setText("The results of your scan(s) will be saved to the default directory")
            self.msg.setInformativeText(str(self.output_path))
            self.msg.setStandardButtons(QMessageBox.Ok) # seperate buttons with "|"
            self.msg.setDefaultButton(QMessageBox.Ok)  # setting default button to Cancel
            self.msg.show()


##    def update_progress(self, progress):
##        progress_dict = progress.format_dict
##        percentage = (progress_dict['n'] + 1) / progress_dict['total'] * 100
##        self.progressBar.setValue(percentage)

    def set_output_paths(self, b):
        """
        Set the output paths for CSVs and HTMLs
        @param b: bool, True if user input an output path, else False.
        """
        if b:
            htmls_path = os.path.join(self.output_dir, 'Plots')        
            csvs_path = os.path.join(self.output_dir, 'CSVs')
        else:
            htmls_path = os.path.join(self.output_path, 'Plots')        
            csvs_path = os.path.join(self.output_path, 'CSVs')

        if not os.path.exists(htmls_path):
          os.mkdir(htmls_path)
        else:
          print("Folder %s already exists" % htmls_path)
        if not os.path.exists(csvs_path):
            os.mkdir(csvs_path)
        else:
          print("Folder %s already exists" % csvs_path)
        return htmls_path, csvs_path


    def get_params(self):
        """
        Stores all the updated user-defined params for clustering algorithms
        """
        for alg in self.checked_cbs:
            if alg == 'DBSCAN':
                self.config['DBSCAN'] = [int(self.p1_photoncount.text()),
                                         int(self.p1_xprecision.text()),
                                         int(self.p1_density_threshold2.text()),
                                         int(self.p1_density_threshold3.text()),
##                                         int(self.p1_min_pts.text()),
                                         int(self.p1_epsilon.text()),
                                         int(self.p1_min_samples.text())]
                if self.p1_upca.isChecked():
                    self.p1_stdev_num = QLineEdit('1.0', self)
                    self.p1_stdev_num.setToolTip('PCA noise reduction standard deviations')
                    self.config['DBSCAN'].append(float(self.p1_stdev_num.text()))

            elif alg == 'HDBSCAN':
                self.config['HDBSCAN'] = [int(self.p2_photoncount.text()),
                                          int(self.p2_xprecision.text()),
                                          int(self.p2_density_threshold2.text()),
                                          int(self.p2_density_threshold3.text()),
                                          int(self.p2_min_cluster_points.text()),
                                          int(self.p2_epsilon.text()),
                                          int(self.p2_min_samples.text()),
                                          str(self.p2_extracting_alg.text()),
                                          float(self.p2_selection_alpha.text())]
                if self.p2_upca.isChecked():
                    self.p2_stdev_num = QLineEdit('1.0', self)
                    self.p2_stdev_num.setToolTip('PCA noise reduction standard deviations')
                    self.config['HDBSCAN'].append(float(self.p2_stdev_num.text()))

            elif alg == 'FOCAL':
                self.config['FOCAL'] = [int(self.p3_photoncount.text()),
                                        int(self.p3_xprecision.text()),
                                        int(self.p3_density_threshold2.text()),
                                        int(self.p3_density_threshold3.text()),
                                        int(self.p3_sigma.text()),
                                        int(self.p3_minL.text()),
                                        int(self.p3_minC.text()),
                                        int(self.p3_minPC.text())] 
                if self.p3_upca.isChecked():
                    self.p3_stdev_num = QLineEdit('1.0', self)
                    self.p3_stdev_num.setToolTip('PCA noise reduction standard deviations')
                    self.config['FOCAL'].append(float(self.p3_stdev_num.text()))
                
        
    def prepare_scan(self):
        '''
        
        '''
        self.btn5 = QPushButton(self)
        self.btn5.setText('Run Scan')
        ##self.btn5.adjustSize()
        self.btn5.clicked.connect(self.run_scan)
        ##self.overall_layout.addWidget(Color('white'), 10, 3)
        self.btn5.setGeometry(30, 40, 200, 25)
        self.overall_layout.addWidget(self.btn5, 10, 0, 1, 5)
        

        # Creating progress bar
##        self.pbar = QProgressBar(self)
##        self.pbar.setValue(0)
##        self.pbar.setGeometry(30, 40, 200, 25)
##        self.overall_layout.addWidget(self.pbar, 15, 0, 1, 5)
        self.pbar = QStatusBar()
##        self.setStatusBar(self.pbar)
##        self.pbar.setGeometry(30, 40, 200, 25)
##        self.overall_layout.addWidget(self.pbar, 15, 0, 1, 5)
        

##    def onStateChanged(self):
##
##        if self.cb1.isChecked():
##            if self.db.isChecked():
####                if self.rn1 == 6:
##                    self.p1_stdev_num = QLineEdit('1.0', self)
##                    self.p1_layout.addRow("PCA Stdev",self.p1_stdev_num)
##
##            if self.hb.isChecked():
####                if self.rn2 == 9:
##                    self.p2_stdev_num = QLineEdit('1.0', self)
##                    self.p2_layout.addRow("PCA Stdev", self.p2_stdev_num)
##
##            if self.fb.isChecked():
####                if self.rn3 == 8:
##                    self.p3_stdev_num = QLineEdit('1.0', self)
##                    self.p3_layout.addRow("PCA Stdev", self.p3_stdev_num)
##
##        else:
##            if self.db.isChecked():
####                self.rn1 = 6
##                self.p1_layout.removeRow(self.p1_stdev_num)
##                self.p1_stdev_num = None
##            if self.hb.isChecked():
####                self.rn2 = 9
##                self.p2_layout.removeRow(self.p2_stdev_num)
##                self.p2_stdev_num = None
##            if self.fb.isChecked():
####                self.rn3 = 8
##                self.p3_layout.removeRow(self.p3_stdev_num)
##                self.p3_stdev_num = None
                

    def onStateChangedDB(self):

        if self.db.isChecked():
            if self.p1_upca.isChecked() & (self.rn1 == 7):
                self.p1_stdev_num = QLineEdit('1.0', self)
                self.p1_layout.addRow("PCA Stdev",self.p1_stdev_num)
                self.rn1 = 8
            else:
                if self.rn1 == 8:
                    self.p1_layout.removeRow(self.p1_stdev_num)
                    self.p1_stdev_num = None
                    self.rn1 = 7

    def onStateChangedHB(self):

        if self.hb.isChecked():
            if self.p2_upca.isChecked() & (self.rn2 == 9):
                self.p2_stdev_num = QLineEdit('1.0', self)
                self.p2_layout.addRow("PCA Stdev", self.p2_stdev_num)
                self.rn2 = 10
            else:
                if self.rn2 == 10:
                    self.p2_layout.removeRow(self.p2_stdev_num)
                    self.p2_stdev_num = None
                    self.rn2 = 9

    def onStateChangedFB(self):

        if self.fb.isChecked():
            if self.p3_upca.isChecked() & (self.rn3 == 8):
                self.p3_stdev_num = QLineEdit('1.0', self)
                self.p3_layout.addRow("PCA Stdev", self.p3_stdev_num)
                self.rn3 = 9
            else:
                if self.rn3 == 9:
                    self.p3_layout.removeRow(self.p3_stdev_num)
                    self.p3_stdev_num = None
                    self.rn3 = 8

    
    def btnstate(self, cb):
        self.num_of_confs = 0
        if cb.text() == 'DBSCAN':
            if cb.isChecked() == True:
                if 'DBSCAN' not in self.checked_cbs:
                    self.checked_cbs.append('DBSCAN')
                    print(self.checked_cbs)
                self.rn1 = 7 # Number of rows for widget1
                self.p1_photoncount = QLineEdit('1000', self)
                self.p1_xprecision = QLineEdit('100', self)
                self.p1_density_threshold2 = QLineEdit('0', self)
                self.p1_density_threshold3 = QLineEdit('0', self)
##                self.p1_min_pts = QLineEdit('22', self)
                self.p1_epsilon = QLineEdit('70', self)
                self.p1_min_samples = QLineEdit('22', self)
                
                self.p1_upca = QCheckBox('')
                self.p1_upca.setChecked(False)
                self.p1_upca.stateChanged.connect(self.onStateChangedDB)

                # Set tips to appear on mouse hover
                self.p1_photoncount.setToolTip('Minimum Photon Count Value to Analyse')
                self.p1_xprecision.setToolTip('Maximum X-Precision Value to Analyse')
                self.p1_density_threshold2.setToolTip('Minimum 2D Density to Consider a Cluster')
                self.p1_density_threshold3.setToolTip('Minimum 3D Density to Consider a Cluster')
                self.p1_min_samples.setToolTip('Minimum Number of Points in Epsilon Radius to Consider a Cluster')
                self.p1_epsilon.setToolTip('DBSCAN Search Radius')
                self.p1_upca.setToolTip('PCA noise reduction standard deviations')
                
                self.p1 = QWidget()
                self.p1_layout = QFormLayout()
                self.p1_label = QLabel("Input DBSCAN Parameters")
                self.p1_layout.addWidget(self.p1_label)
                self.setLayout(self.p1_layout)
##                self.p1_layout.addRow("MinPts", self.p1_min_pts)
                self.p1_layout.addRow("Epsilon", self.p1_epsilon)
                self.p1_layout.addRow("MinSamples", self.p1_min_samples)
                self.p1_layout.addRow("2D Density Threshold",self.p1_density_threshold2)
                self.p1_layout.addRow("3D Density Threshold",self.p1_density_threshold3)
                self.p1_layout.addRow("Min Photon-count",self.p1_photoncount)
                self.p1_layout.addRow("Max X-Precision", self.p1_xprecision)
                self.p1_layout.addRow("Use PCA", self.p1_upca)                
                
                self.p1.setLayout(self.p1_layout)
                ##self.overall_layout.addWidget(Color('white'), 3, 0, rn1, 1)
                ##self.overall_layout.addWidget(self.p1, 3, 0, rn1, 1)
##                self.overall_layout.addWidget(Color('white'), 3, 0)
                self.overall_layout.addWidget(self.p1, 3, 0)
                
                self.algs[0][1] = True
                self.num_of_confs += 1

            else:
                self.p1.setVisible(False)
                self.algs[0][1] = False
                self.num_of_confs -= 1
##                del self.config['DBSCAN']
                if 'DBSCAN' in self.checked_cbs:
                    self.checked_cbs.remove('DBSCAN')
                    print(self.checked_cbs)
                
                
        elif cb.text() == 'HDBSCAN':
            if cb.isChecked() == True:
                if 'HDBSCAN' not in self.checked_cbs:
                    self.checked_cbs.append('HDBSCAN')
                    print(self.checked_cbs)
                self.rn2 = 9
                self.p2_photoncount = QLineEdit('1000', self)
                self.p2_xprecision = QLineEdit('100', self)
                self.p2_density_threshold2 = QLineEdit('0', self)
                self.p2_density_threshold3 = QLineEdit('0', self)
                self.p2_min_cluster_points = QLineEdit('15', self)
                self.p2_epsilon = QLineEdit('70', self)
                self.p2_min_samples = QLineEdit('22', self)
                self.p2_extracting_alg = QLineEdit('leaf', self)
                self.p2_selection_alpha = QLineEdit('1.0', self)
                self.p2_upca = QCheckBox('')
                self.p2_upca.setChecked(False)
                self.p2_upca.stateChanged.connect(self.onStateChangedHB)

                # Set tips to appear on mouse hover
                self.p2_photoncount.setToolTip('Minimum Photon Count Value to Analyse')
                self.p2_xprecision.setToolTip('Maximum X-Precision Value to Analyse')
                self.p2_density_threshold2.setToolTip('Minimum 2D density to consider a cluster')
                self.p2_density_threshold3.setToolTip('Minimum 3D density to consider a cluster')
                self.p2_min_cluster_points.setToolTip('The smallest size grouping to consider a cluster')
                self.p2_epsilon.setToolTip('HDBSCAN minimum cluster radius')
                self.p2_min_samples.setToolTip('A measure of how conservative the clustering is.\nThe larger the value,\nmore points will be declared as noise,\nand clusters will be restricted to progressively denser areas.')
                self.p2_extracting_alg.setToolTip('Determines how HDBSCAN selects flat clusters from the cluster tree hierarchy\nOptions are: eom or leaf')
                self.p2_selection_alpha.setToolTip('I suggest not to change this parameter!\nIncreasing alpha will make the clustering more conservative,\nbut on a much tighter scale')
                self.p2_upca.setToolTip('PCA noise reduction standard deviations')
                
                self.p2 = QWidget()
                self.p2_layout = QFormLayout()
                self.p2_label = QLabel("Input HDBSCAN Parameters")
                self.p2_layout.addWidget(self.p2_label)
                self.setLayout(self.p2_layout)
                self.p2_layout.addRow("MinPts", self.p2_min_cluster_points)
                self.p2_layout.addRow("Epsilon", self.p2_epsilon)
                self.p2_layout.addRow("MinSamples", self.p2_min_samples)
                self.p2_layout.addRow("Extracting Algorithm",self.p2_extracting_alg)
                self.p2_layout.addRow("Selection Alpha", self.p2_selection_alpha)
                self.p2_layout.addRow("2D Density Threshold",self.p2_density_threshold2)
                self.p2_layout.addRow("3D Density Threshold",self.p2_density_threshold3)
                self.p2_layout.addRow("Min Photon-count",self.p2_photoncount)
                self.p2_layout.addRow("Max X-Precision", self.p2_xprecision)
                self.p2_layout.addRow("Use PCA", self.p2_upca)

                self.p2.setLayout(self.p2_layout)
##                self.overall_layout.addWidget(Color('white'), 3, 1, rn2, 1)
##                self.overall_layout.addWidget(self.p2, 3, 1, rn2, 1)
##                self.overall_layout.addWidget(Color('white'), 3, 1)
                self.overall_layout.addWidget(self.p2, 3, 1)
                
                self.algs[1][1] = True
                self.num_of_confs += 1

            else:
                self.p2.setVisible(False)
                self.algs[1][1] = False
                self.num_of_confs -= 1
##                del self.config['HDBSCAN']
                if 'HDBSCAN' in self.checked_cbs:
                    self.checked_cbs.remove('HDBSCAN')
                    print(self.checked_cbs)

        if cb.text() == 'FOCAL':
            if cb.isChecked() == True:
                if 'FOCAL' not in self.checked_cbs:
                    self.checked_cbs.append('FOCAL')
                    print(self.checked_cbs)
                self.rn3 = 8
                self.p3_photoncount = QLineEdit('1000', self)
                self.p3_xprecision = QLineEdit('100', self)
                self.p3_density_threshold2 = QLineEdit('0', self)
                self.p3_density_threshold3 = QLineEdit('0', self)
                self.p3_sigma = QLineEdit('55', self)
                self.p3_minL = QLineEdit('1', self)
                self.p3_minC = QLineEdit('25', self)
                self.p3_minPC = QLineEdit('1500', self)
                self.p3_upca = QCheckBox('')
                self.p3_upca.setChecked(False)
                self.p3_upca.stateChanged.connect(self.onStateChangedFB)
##                self.config['FOCAL'] = [int(self.p3_photoncount.text()),
##                                        int(self.p3_xprecision.text()),
##                                        int(self.p3_density_threshold2.text()),
##                                        int(self.p3_density_threshold3.text()),
##                                        int(self.p3_sigma.text()),
##                                        int(self.p3_minL.text()),
##                                        int(self.p3_minC.text()),
##                                        int(self.p3_minPC.text())]

                self.p3_photoncount.setToolTip('Minimum Photon Count Value to Analyse')
                self.p3_xprecision.setToolTip('Maximum X-Precision Value to Analyse')
                self.p3_density_threshold2.setToolTip('Minimum 2D density to consider a cluster')
                self.p3_density_threshold3.setToolTip('Minimum 3D density to consider a cluster')
                self.p3_sigma.setToolTip('Voxel size')
                self.p3_minL.setToolTip('Minimum threshold for voxel neighbour score')
                self.p3_minC.setToolTip('Minimum threshold for number of voxels per cluster')
                self.p3_upca.setToolTip('PCA noise reduction standard deviations')

                self.p3 = QWidget()
                self.p3_layout = QFormLayout()
                self.p3_label = QLabel("Input FOCAL Parameters")
                self.p3_layout.addWidget(self.p3_label)
                self.setLayout(self.p3_layout)

                self.p3_layout.addRow("Grid Size", self.p3_sigma)
                self.p3_layout.addRow("minL", self.p3_minL)
                self.p3_layout.addRow("minC", self.p3_minC)
                self.p3_layout.addRow("2D Density Threshold",self.p3_density_threshold2)
                self.p3_layout.addRow("3D Density Threshold",self.p3_density_threshold3)
                self.p3_layout.addRow("Min Photon-count",self.p3_photoncount)
                self.p3_layout.addRow("Max X-Precision", self.p3_xprecision)
                self.p3_layout.addRow("Min Average PhotonCount", self.p3_minPC)
                self.p3_layout.addRow("Use PCA", self.p3_upca)
                
                self.p3.setLayout(self.p3_layout)
##                self.overall_layout.addWidget(Color('white'), 3, 2, rn3, 1)
##                self.overall_layout.addWidget(self.p3, 3, 2, rn3, 1)
##                self.overall_layout.addWidget(Color('white'), 3, 2)
                self.overall_layout.addWidget(self.p3, 3, 2)
                
                self.algs[2][1] = True
                self.num_of_confs += 1

            else:
                self.p3.setVisible(False)
                self.algs[2][1] = False
                self.num_of_confs -= 1
##                del self.config['FOCAL']
                if 'FOCAL' in self.checked_cbs:
                    self.checked_cbs.remove('FOCAL')
                print(self.checked_cbs)
        
        if cb.isChecked() == True:
            self.prepare_scan()
        
    
    def clickable(self, widget):
        """ class taken from 
        https://wiki.python.org/moin/PyQt/Making%20non-clickable%20widgets%20clickable """
        class Filter(QObject):
            clicked = pyqtSignal()
            def eventFilter(self, obj, event):
                if obj == widget:
                    if event.type() == QEvent.MouseButtonRelease:
                        if obj.rect().contains(event.pos()):
                            self.clicked.emit()
                            # The developer can opt for .emit(obj) to get the object within the slot.
                            return True
                return False
        filter = Filter(widget)
        widget.installEventFilter(filter)
        return filter.clicked

    def show_files(self):
        """
        Creates a list widget on the main window that shows a list of all uploaded files to select for analysis
        """
        self.selected_files = []
        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(2)
##        self.listwidget.itemSelectionChanged.connect(self.on_change)
        self.listwidget.clicked.connect(self.clicked)

        for i,fpath in enumerate(self.fileslist):
            filename = os.path.basename(fpath)
            self.listwidget.insertItem(i, str(filename))
        
##        self.listwidget.clicked.connect(self.clicked)
##        self.listwidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
##        self.listwidget.setSelectionMode(3)
##        self.listwidget.setToolTip('Select files for analysis,\nTo select multiple files hold Shift/CTL/CMD')
        self.listwidget.adjustSize()
        self.overall_layout.addWidget(self.listwidget, 1, 3, 3, 2)

        self.sel_all_btn = QPushButton(self)
        self.sel_all_btn.setText('Select All')
        self.sel_all_btn.adjustSize()
        self.sel_all_btn.setToolTip('Select all file(s) for analysis')
        self.sel_all_btn.clicked.connect(self.select_all_files)
        self.overall_layout.addWidget(self.sel_all_btn, 4, 3, 1, 2)

        self.deselect_btn = QPushButton(self)
        self.deselect_btn.setText('Deselect All')
        self.deselect_btn.adjustSize()
        self.deselect_btn.setToolTip('Deselect all file(s) for analysis')
        self.deselect_btn.clicked.connect(self.deselect_all_files)
        self.overall_layout.addWidget(self.deselect_btn, 5, 3, 1, 2)
        
        self.del_btn = QPushButton(self)
        self.del_btn.setText('Remove file(s)')
        self.del_btn.adjustSize()
        self.del_btn.setToolTip('Remove selected file(s) from list')
        self.del_btn.clicked.connect(self.remove_files)
        self.overall_layout.addWidget(self.del_btn, 6, 3, 1, 2)


    def select_all_files(self):
        """
        Selects all uploaded files for analysis
        """
        for index in range(self.listwidget.count()): # Iterate through all items in the list
            self.listwidget.item(index).setSelected(True)
            t = self.listwidget.item(index).text()
            if t not in self.selected_files:
                self.selected_files.append(t)
                print(t, ' has been added to selected files list')
        if len(self.selected_files) >= 1:
            self.db.setCheckable(True)
            self.hb.setCheckable(True)
            self.fb.setCheckable(True)
            


    def deselect_all_files(self):
        """
        Deselects all files in the uploaded files list, but keeps them available
        """
        for index in range(self.listwidget.count()):
            self.listwidget.item(index).setSelected(False)
            t = self.listwidget.item(index).text()
            if t in self.selected_files:
                self.selected_files.remove(t)
                print(t, ' has been removed from selected files list')
        if len(self.selected_files) < 1:
            self.db.setCheckable(False)
            self.hb.setCheckable(False)
            self.fb.setCheckable(False)
            
    

    def remove_files(self):
        """
        Removes selected files from files list
        """
        item = self.listwidget.currentItem()
        t = item.text()
        if t in self.selected_files:
            self.selected_files.remove(t)
            print(t, ' has been removed from selected files')
        self.listwidget.takeItem(self.listwidget.currentRow())
        print(t, ' has been removed from list')
        
        
    def get_files(self):
        """
        Opens multiple files to scan
        """
        self.fnames = QFileDialog.getOpenFileNames(
            parent = self,
            caption = 'Open files',
            filter = 'Loc files (*.csv *.xlsx)'
            )
        filenames = []
        for i, filename in enumerate(self.fnames[0]):
            if filename.endswith((".csv", ".xlsx")):
                filenames.append(filename)
        self.path = ['files', self.fnames]
        self.fileslist = filenames
        self.show_files()


    def get_dir(self):
        """
        Opens all files in a directory
        """
        self.dr = QFileDialog.getExistingDirectory(
            parent = self,
            caption = 'Select a Folder'
            )

        filenames = []
        if os.path.isdir(self.dr):
            for filename in os.listdir(self.dr):
                if filename.endswith((".csv", ".xlsx")):
                    filenames.append(filename)
        self.path = ['dir', self.dr]
        self.fileslist = filenames
        self.show_files()

    def get_output_dir(self):
        """
        Gets a path to a directory and saves it as a string
        """
        self.output_dir = QFileDialog.getExistingDirectory(
            parent = self,
            caption = 'Select a Folder'
            )
        self.outdir = QLabel(str(self.output_dir))
        self.output_dir_set = True
        self.overall_layout.addWidget(self.outdir, 6, 1, 1, 2)

    def checked_show_plots(self):
        """
        Tells the plotting function to open plots when they're ready, and not only save them
        """
        if self.show_plots_cb.isChecked():
            self.open_plots = True
        else:
            self.open_plots = False


##    def on_change(self):
##
##        for item in self.listwidget.selectedItems():
##            t = item.text()
##            if t not in self.selected_files:
##                self.selected_files.append(t)
##                print(t, ' has been added to selected files list')
##            else:
##                self.selected_files.remove(t)
##                print(t, ' has been removed from selected files list')
##        print('Selected Files List:\n', self.selected_files)
                
            

    def clicked(self, qmodelindex):
        item = self.listwidget.currentItem()
        t = item.text()
        if t not in self.selected_files:
            self.selected_files.append(t)
            print(t, ' has been added to selected files list')
        else:
            self.selected_files.remove(t)
            print(t, ' has been removed from selected files list')
        if len(self.selected_files) >= 1:
            self.db.setCheckable(True)
            self.hb.setCheckable(True)
            self.fb.setCheckable(True)
        else:
            self.db.setCheckable(False)
            self.hb.setCheckable(False)
            self.fb.setCheckable(False)
    
        

    def close_window(self):
        self.param_box.close()
        self.param_box = None



if __name__ == '__main__':
    app = QApplication([])
##    app.setStyleSheet('QLabel{color: white;}')
    win = MainWindow()
    win.show()

    app.exec_() #Kickstart the Qt event loop



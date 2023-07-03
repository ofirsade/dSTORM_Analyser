
from PyQt5.QtWidgets import *
"""QApplication, QMainWindow, QLabel, QGridLayout, \
  QWidget, QPushButton, QTableWidget, QTableWidgetItem, QInputDialog, \
  QMessageBox, QComboBox, QLineEdit, QAction, QCheckBox, QFileDialog, QVBoxLayout"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import * #QIcon
from PyQt5.QtCore import * #pyqtSlot
import pyqtgraph as pg
import sys, os
from os import listdir, path, walk, mkdir, unlink, environ
from shutil import rmtree, copyfile
from datetime import datetime
import time
import re
from utils.Prescan import dstorm_dataset
import multiprocessing

##if getattr(sys, 'frozen', False):
##    os.environ['JOBLIB_MULTIPROCESSING'] = '0'

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
        self.output_dir = os.path.abspath(os.path.dirname(__file__))
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

        self.gf = False
        
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

        self.cb1 = QCheckBox('Use PCA')
        self.cb1.setChecked(False)
##        self.cb1.stateChanged.connect(lambda:self.use_PCA(self.cb1))
        self.overall_layout.addWidget(self.cb1, 1, 2)        

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
        if self.output_dir != os.path.abspath(os.path.dirname(__file__)):
            self.dataset = dstorm_dataset(self.path, self.output_dir, self.selected_files, self.algs, self.config, self.pbar)
##            if self.dataset.completed == True:
##                self.msg = QMessageBox()
##                self.msg.setWindowTitle("Scan is Completed")
##                self.msg.setStandardButtons(QMessageBox.Ok) # seperate buttons with "|"
##                self.msg.setDefaultButton(QMessageBox.Ok)  # setting default button to Ok
##                self.msg.show()

##        elif self.prep_data == 0:
##            self.prep_data += 1
##            self.msg = QMessageBox()
##            self.msg.setWindowTitle("Warning")
##            self.msg.setText("You have not selected an output path!")
##            self.msg.setInformativeText("To run without selecting a path, click Ok and run again")
##            self.msg.setIcon(QMessageBox.Warning)
##            self.msg.setStandardButtons(QMessageBox.Ok) # seperate buttons with "|"
##            self.msg.setDefaultButton(QMessageBox.Ok)  # setting default button to Ok
##            self.msg.show()
        else:
            output_path = os.path.join(self.output_dir, "Results")
            self.msg = QMessageBox()
            self.msg.setText("The results of your scan(s) will be saved to the default directory")
            self.msg.setInformativeText(str(output_path))
            self.msg.setStandardButtons(QMessageBox.Ok) # seperate buttons with "|"
            self.msg.setDefaultButton(QMessageBox.Ok)  # setting default button to Cancel
            self.msg.show()
            self.dataset = dstorm_dataset(self.path, output_path, self.selected_files, self.algs, self.config, self.pbar)
##            output_path = os.path.join(self.output_dir, "Results")
##            self.dataset = dstorm_dataset(self.path, output_path, self.selected_files, self.algs, self.config, self.pbar)
####            if self.dataset.completed == True:
##                self.msg = QMessageBox()
##                self.msg.setWindowTitle("Scan is Completed")
##                self.msg.setStandardButtons(QMessageBox.Ok) # seperate buttons with "|"
##                self.msg.setDefaultButton(QMessageBox.Ok)  # setting default button to Ok
##                self.msg.show()


    def get_params(self):
        """
        Stores all the updated user-defined params
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
                if self.cb1.isChecked() == True:
                    self.p1_stdev_num = QLineEdit('1.0', self)
                    self.config['DBSCAN'].append(float(self.p1_stdev_num.text()))
                    self.p1_layout.addRow("PCA Stdev",self.p1_stdev_num)

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
                if self.cb1.isChecked() == True:
                    self.p2_stdev_num = QLineEdit('1.0', self)
                    self.config['HDBSCAN'].append(float(self.p2_stdev_num.text()))
                    self.p2_layout.addRow("PCA Stdev",self.p2_stdev_num)

            elif alg == 'FOCAL':
                self.config['FOCAL'] = [int(self.p3_photoncount.text()),
                                        int(self.p3_xprecision.text()),
                                        int(self.p3_density_threshold2.text()),
                                        int(self.p3_density_threshold3.text()),
                                        int(self.p3_sigma.text()),
                                        int(self.p3_minL.text()),
                                        int(self.p3_minC.text()),
                                        int(self.p3_minPC.text())] 
                if self.cb1.isChecked() == True:
                    self.p3_stdev_num = QLineEdit('1.0', self)
                    self.config['FOCAL'].append(float(self.p3_stdev_num.text()))
                    self.p3_layout.addRow("PCA Stdev",self.p3_stdev_num)
                
        
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
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)
        self.overall_layout.addWidget(self.pbar, 15, 0, 1, 5)


        
    def btnstate(self, cb):
        self.num_of_confs = 0
        if cb.text() == 'DBSCAN':
            if cb.isChecked() == True:
                if 'DBSCAN' not in self.checked_cbs:
                    self.checked_cbs.append('DBSCAN')
                rn1 = 6 # Number of rows for widget
                self.p1_photoncount = QLineEdit('1000', self)
                self.p1_xprecision = QLineEdit('100', self)
                self.p1_density_threshold2 = QLineEdit('0', self)
                self.p1_density_threshold3 = QLineEdit('0', self)
##                self.p1_min_pts = QLineEdit('22', self)
                self.p1_epsilon = QLineEdit('70', self)
                self.p1_min_samples = QLineEdit('22', self)
                
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
                
                if self.cb1.isChecked() == True:
                    self.p1_stdev_num = QLineEdit('1.0', self)
##                    self.config['DBSCAN'].append(float(self.p1_stdev_num.text()))
                    self.p1_layout.addRow("PCA Stdev",self.p1_stdev_num)
                    rn1 = 7
                
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
                rn2 = 9
                self.p2_photoncount = QLineEdit('1000', self)
                self.p2_xprecision = QLineEdit('100', self)
                self.p2_density_threshold2 = QLineEdit('0', self)
                self.p2_density_threshold3 = QLineEdit('0', self)
                self.p2_min_cluster_points = QLineEdit('15', self)
                self.p2_epsilon = QLineEdit('70', self)
                self.p2_min_samples = QLineEdit('22', self)
                self.p2_extracting_alg = QLineEdit('leaf', self)
                self.p2_selection_alpha = QLineEdit('1.0', self)
##                self.config['HDBSCAN'] = [int(self.p2_photoncount.text()),
##                                          int(self.p2_xprecision.text()),
##                                          int(self.p2_density_threshold2.text()),
##                                          int(self.p2_density_threshold3.text()),
##                                          int(self.p2_min_pts.text()),
##                                          int(self.p2_epsilon.text()),
##                                          int(self.p2_min_samples.text()),
##                                          str(self.p2_extracting_alg.text()),
##                                          float(self.p2_selection_alpha.text())]
                
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

                if self.cb1.isChecked() == True:
                    self.p2_stdev_num = QLineEdit('1.0', self)
##                    self.config['HDBSCAN'].append(float(self.p2_stdev_num.text()))
                    self.p2_layout.addRow("PCA Stdev",self.p2_stdev_num)
                    rn2 = 10
                
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

        if cb.text() == 'FOCAL':
            if cb.isChecked() == True:
                if 'FOCAL' not in self.checked_cbs:
                    self.checked_cbs.append('FOCAL')
                    print(self.checked_cbs)
                rn3 = 8
                self.p3_photoncount = QLineEdit('1000', self)
                self.p3_xprecision = QLineEdit('100', self)
                self.p3_density_threshold2 = QLineEdit('0', self)
                self.p3_density_threshold3 = QLineEdit('0', self)
                self.p3_sigma = QLineEdit('55', self)
                self.p3_minL = QLineEdit('1', self)
                self.p3_minC = QLineEdit('25', self)
                self.p3_minPC = QLineEdit('2000', self)
##                self.config['FOCAL'] = [int(self.p3_photoncount.text()),
##                                        int(self.p3_xprecision.text()),
##                                        int(self.p3_density_threshold2.text()),
##                                        int(self.p3_density_threshold3.text()),
##                                        int(self.p3_sigma.text()),
##                                        int(self.p3_minL.text()),
##                                        int(self.p3_minC.text()),
##                                        int(self.p3_minPC.text())]              

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
                if self.cb1.isChecked() == True:
                    self.p3_stdev_num = QLineEdit('1.0', self)
##                    self.config['FOCAL'].append(float(self.p3_stdev_num.text()))
                    self.p3_layout.addRow("PCA Stdev",self.p3_stdev_num)
                    rn3 = 9
                
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

        self.del_btn = QPushButton(self)
        self.del_btn.setText('Remove file(s)')
        self.del_btn.adjustSize()
        self.del_btn.setToolTip('Remove selected file(s) from list')
        self.del_btn.clicked.connect(self.remove_files)
        self.overall_layout.addWidget(self.del_btn, 4, 3, 1, 2)
        

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
                    print(filename)
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
        self.overall_layout.addWidget(self.outdir, 6, 1, 1, 2)


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
        

    def use_PCA(self, cb):

        if cb.isChecked() == True:
            ### Use PCA Noise Reduction (default --> no change)
            pass
        else:
            ### Do not use PCA Noise Reduction
            pass
    

##    def show_in_window(fig):
##    
##        plotly.offline.plot(fig, filename='name.html', auto_open=False)
##        
##        app = QApplication(sys.argv)
##        web = QWebEngineView()
##        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "name.html"))
##        web.load(QUrl.fromLocalFile(file_path))
##        web.show()
##        sys.exit(app.exec_())
##
##
##    def plot_res(xyzl, fname, unique_labels, alg, output_path):
##
##        fig = make_subplots(
##            rows = 1, 
##            cols = 1, 
##    ##        vertical_spacing=0.05,
##            subplot_titles=orig_df.filename.to_list()
##        )
##
##        
##        bg_df = xyzl.loc[xyzl['Label'] == -1]
##        clustered_df = xyzl.loc[xyzl['Label'] != -1]
##        # Probe 0
##        fig.add_trace(
##            go.Scattergl(
##                x = bg_df['x'].values,
##                y = bg_df['y'].values,
##                mode = 'markers',
##                marker=dict(
##                    color='grey',           # set color to an array/list of desired values
##                    opacity=0.1
##                )
##            ),
##            row = 1, col = 1
##        )
##            
##        # Draw clusters
##        for i in unique_labels:
##            pc = clustered_df.loc[clustered_df['Label'] == i]
##            
##            fig.add_trace(
##                go.Scattergl(
##                    x = pc['x'].values,
##                    y = pc['y'].values,
##                    mode = 'markers',
##                    marker = dict(
##                        color = i, #cmap(np.sqrt(colocalization[i])),    
##                        colorscale = 'rainbow',
##                        opacity = 0.5
##                    )
##                ),
##                row = 1, col = 1
##            )
##        
##        fig.update_xaxes(range = [0, 18e3], row = 1, col = 1)     
##        fig.update_yaxes(scaleanchor = 'x', scaleratio = 1, row = 1, col = 1)
##
##        show_in_window(fig)
        

    def close_window(self):
        self.param_box.close()
        self.param_box = None



if __name__ == '__main__':
    app = QApplication([])
##    app.setStyleSheet('QLabel{color: white;}')
    win = MainWindow()
    win.show()

    app.exec_() #Kickstart the Qt event loop

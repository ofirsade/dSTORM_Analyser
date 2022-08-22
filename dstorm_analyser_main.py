
from PyQt5.QtWidgets import *
"""QApplication, QMainWindow, QLabel, QGridLayout, \
  QWidget, QPushButton, QTableWidget, QTableWidgetItem, QInputDialog, \
  QMessageBox, QComboBox, QLineEdit, QAction, QCheckBox, QFileDialog, QVBoxLayout"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import * #QIcon
from PyQt5.QtCore import * #pyqtSlot
import pyqtgraph as pg
import sys
from os import listdir, path, walk, mkdir, unlink, environ
from shutil import rmtree, copyfile
from datetime import datetime


#from utils.Prescan import prep_data
from utils.Prescan import dstorm_dataset


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
    
    def init_UI(self):

        self.resize(1000, 800)
        self.setWindowTitle('dSTORM Analyser')
        
        self.main_layout = QGridLayout()
        
        self.label = QLabel(self)
        self.label.setText("Welcome to dSTORM Analyser")
        my_font = QtGui.QFont('Arial', 30)
        my_font.setBold(True)
        self.label.setFont(QtGui.QFont(my_font))
        self.label.adjustSize()
        self.label.setAlignment(Qt.AlignCenter)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.overall_layout = QGridLayout(central_widget)
        self.overall_layout.addWidget(Color('white'), 0, 0, 4, 7)

        # start at row 0, column 0, use 1 row and 3 columns. 
        self.overall_layout.addWidget(self.label, 0, 0, 1, 3)
        
        self.btn1 = QPushButton(self)
        self.btn1.setText('Upload File')
        self.btn1.adjustSize()
        self.btn1.clicked.connect(self.get_file)
        self.overall_layout.addWidget(self.btn1, 1, 0)

        self.btn2 = QPushButton(self)
        self.btn2.setText('Upload Multiple Files')
        self.btn2.adjustSize()
        self.btn2.clicked.connect(self.get_files)
        self.overall_layout.addWidget(self.btn2, 1, 1)

        self.btn3 = QPushButton(self)
        self.btn3.setText('Select Directory')
        self.btn3.adjustSize()
        self.btn3.clicked.connect(self.get_dir)
        self.overall_layout.addWidget(self.btn3, 1, 2)

        self.cb1 = QCheckBox('Use PCA')
        self.cb1.setChecked(True)
        self.cb1.stateChanged.connect(lambda:self.use_PCA(self.cb1))
        self.overall_layout.addWidget(self.cb1, 3, 3)        

        self.db = QCheckBox('DBSCAN')
        self.db.setChecked(False)
        self.db.stateChanged.connect(lambda:self.btnstate(self.db))
        self.overall_layout.addWidget(self.db, 3, 0)

        self.hb = QCheckBox('HDBSCAN')
        self.hb.setChecked(False)
        self.hb.stateChanged.connect(lambda:self.btnstate(self.hb))
        self.overall_layout.addWidget(self.hb, 3, 1)

        self.fb = QCheckBox('FOCAL')
        self.fb.setChecked(False)
        self.fb.stateChanged.connect(lambda:self.btnstate(self.fb))
        self.overall_layout.addWidget(self.fb, 3, 2)

        self.setLayout(self.overall_layout)


    def prepare_data(self):
        """
        Get all relevant data from file(s) input by the user
        """
        self.dataset = dstorm_dataset(self.path, self.algs, self.config)

        
    def run_scan(self):
        self.btn5 = QPushButton(self)
        self.btn5.setText('Run Scan')
        self.btn5.adjustSize()
        self.btn5.clicked.connect(self.prepare_data)
        self.overall_layout.addWidget(Color('white'), 10, 3)
        self.overall_layout.addWidget(self.btn5, 10, 3)
        #res_window = Window()

        
    def btnstate(self, cb):
        self.num_of_confs = 0
        if cb.text() == 'DBSCAN':
            if cb.isChecked() == True:
                rn = 6 # Number of rows for widget
                self.p1_photoncount = QLineEdit('1000', self)
                self.p1_density_threshold2 = QLineEdit('0', self)
                self.p1_density_threshold3 = QLineEdit('0', self)
                self.p1_min_pts = QLineEdit('0', self)
                self.p1_epsilon = QLineEdit('70', self)
                self.p1_min_samples = QLineEdit('22', self)
                self.config['DBSCAN'] = [int(self.p1_photoncount.text()),
                                         int(self.p1_density_threshold2.text()),
                                         int(self.p1_density_threshold3.text()),
                                         int(self.p1_min_pts.text()),
                                         int(self.p1_epsilon.text()),
                                         int(self.p1_min_samples.text())]
                
                self.p1 = QWidget()
                self.p1_layout = QFormLayout()
                self.p1_label = QLabel("Input DBSCAN Parameters")
                self.p1_layout.addWidget(self.p1_label)
                self.setLayout(self.p1_layout)
                self.p1_layout.addRow("MinPts", self.p1_min_pts)
                self.p1_layout.addRow("Epsilon", self.p1_epsilon)
                self.p1_layout.addRow("MinSamples", self.p1_min_samples)
                self.p1_layout.addRow("2D Density Threshold",self.p1_density_threshold2)
                self.p1_layout.addRow("3D Density Threshold",self.p1_density_threshold3)
                self.p1_layout.addRow("Photon-count",self.p1_photoncount)
                
                if self.cb1.isChecked() == True:
                    self.p1_stdev_num = QLineEdit('1.0', self)
                    self.config['DBSCAN'].append(float(self.p1_stdev_num.text()))
                    self.p1_layout.addRow("PCA Stdev",self.p1_stdev_num)
                    rn = 7
                
                self.p1.setLayout(self.p1_layout)
                self.overall_layout.addWidget(Color('white'), 4, 0, rn, 1)
                self.overall_layout.addWidget(self.p1, 4, 0, rn, 1)
                
                self.algs[0][1] = True
                self.num_of_confs += 1
            else:
                self.p1.setVisible(False)
                self.algs[0][1] = False
                self.num_of_confs -= 1
                del self.config['DBSCAN']
                
        elif cb.text() == 'HDBSCAN':
            if cb.isChecked() == True:
                rn = 8
                self.p2_photoncount = QLineEdit('1000', self)
                self.p2_density_threshold2 = QLineEdit('0', self)
                self.p2_density_threshold3 = QLineEdit('0', self)
                self.p2_min_pts = QLineEdit('50', self)
                self.p2_epsilon = QLineEdit('-9999', self)
                self.p2_min_samples = QLineEdit('1', self)
                self.p2_extracting_alg = QLineEdit('leaf', self)
                self.p2_selection_alpha = QLineEdit('1.0', self)
                self.config['HDBSCAN'] = [int(self.p2_photoncount.text()),
                                          int(self.p2_density_threshold2.text()),
                                          int(self.p2_density_threshold3.text()),
                                          int(self.p2_min_pts.text()),
                                          int(self.p2_epsilon.text()),
                                          int(self.p2_min_samples.text()),
                                          str(self.p2_extracting_alg.text()),
                                          float(self.p2_selection_alpha.text())]
                
                self.p2 = QWidget()
                self.p2_layout = QFormLayout()
                self.p2_label = QLabel("Input HDBSCAN Parameters")
                self.p2_layout.addWidget(self.p2_label)
                self.setLayout(self.p2_layout)
                self.p2_layout.addRow("MinPts", self.p2_min_pts)
                self.p2_layout.addRow("Epsilon", self.p2_epsilon)
                self.p2_layout.addRow("MinSamples", self.p2_min_samples)
                self.p2_layout.addRow("Extracting Algorithm",self.p2_extracting_alg)
                self.p2_layout.addRow("Selection Alpha", self.p2_selection_alpha)
                self.p2_layout.addRow("2D Density Threshold",self.p2_density_threshold2)
                self.p2_layout.addRow("3D Density Threshold",self.p2_density_threshold3)
                self.p2_layout.addRow("Photon-count",self.p2_photoncount)
                if self.cb1.isChecked() == True:
                    self.p2_stdev_num = QLineEdit('1.0', self)
                    self.config['HDBSCAN'].append(float(self.p2_stdev_num.text()))
                    self.p2_layout.addRow("PCA Stdev",self.p2_stdev_num)
                    rn = 9
                
                self.p2.setLayout(self.p2_layout)
                self.overall_layout.addWidget(Color('white'), 4, 1, rn, 1)
                self.overall_layout.addWidget(self.p2, 4, 1, rn, 1)
                
                self.algs[1][1] = True
                self.num_of_confs += 1

            else:
                self.p2.setVisible(False)
                self.algs[1][1] = False
                self.num_of_confs -= 1
                del self.config['HDBSCAN']

        if cb.text() == 'FOCAL':
            if cb.isChecked() == True:
                rn = 6
                self.p3_photoncount = QLineEdit('1000', self)
                self.p3_density_threshold2 = QLineEdit('0', self)
                self.p3_density_threshold3 = QLineEdit('0', self)
                self.p3_sigma = QLineEdit('55', self)
                self.p3_minL = QLineEdit('1', self)
                self.p3_minC = QLineEdit('25', self)
                self.config['FOCAL'] = [int(self.p3_photoncount.text()),
                                        int(self.p3_density_threshold2.text()),
                                        int(self.p3_density_threshold3.text()),
                                        int(self.p3_sigma.text()),
                                        int(self.p3_minL.text()),
                                        int(self.p3_minC.text())]              

                self.p3 = QWidget()
                self.p3_layout = QFormLayout()
                self.p3_label = QLabel("Input FOCAL Parameters")
                self.p3_layout.addWidget(self.p3_label)
                self.setLayout(self.p3_layout)

                self.p3_layout.addRow("Grid Size", self.p3_sigma)
                self.p3_layout.addRow("Epsilon", self.p3_minL)
                self.p3_layout.addRow("MinSamples", self.p3_minC)
                self.p3_layout.addRow("2D Density Threshold",self.p3_density_threshold2)
                self.p3_layout.addRow("3D Density Threshold",self.p3_density_threshold3)
                self.p3_layout.addRow("Photon-count",self.p3_photoncount)
                if self.cb1.isChecked() == True:
                    self.p3_stdev_num = QLineEdit('1.0', self)
                    self.config['FOCAL'].append(float(self.p3_stdev_num.text()))
                    self.p3_layout.addRow("PCA Stdev",self.p3_stdev_num)
                    rn = 7
                
                self.p3.setLayout(self.p3_layout)
                self.overall_layout.addWidget(Color('white'), 4, 2, rn, 1)
                self.overall_layout.addWidget(self.p3, 4, 2, rn, 1)
                
                self.algs[2][1] = True
                self.num_of_confs += 1

            else:
                self.p3.setVisible(False)
                self.algs[2][1] = False
                self.num_of_confs -= 1
                del self.config['FOCAL']
        
        if cb.isChecked() == True:
            self.run_scan()
        
            

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

    
    def get_file(self):
        """
        Opens a single file to scan
        """
        self.fname = QFileDialog.getOpenFileName(
            parent = self,
            caption = 'Select a .csv or .xlsx file',
            filter = "Loc files (*.csv *.xlsx)"
            )
        self.path = ['file', self.fname]
        

    def get_files(self):
        """
        Opens multiple files to scan
        """
        self.fnames = QFileDialog.getOpenFileNames(
            parent = self,
            caption = 'Open files',
            filter = 'Loc files (*.csv *.xlsx)'
            )
        self.path = ['files', self.fnames]

    def get_dir(self):
        """
        Opens all files in a directory
        """
        self.dr = QFileDialog.getExistingDirectory(
            parent = self,
            caption = 'Select a Folder'
            )
        self.path = ['dir', self.dr]

    def use_PCA(self, cb):

        if cb.isChecked() == True:
            ### Use PCA Noise Reduction (default --> no change)
            pass
        else:
            ### Do not use PCA Noise Reduction
            pass
    
    def run_DBSCAN(self):
        min_pts_val = float(self.p1_min_pts.text())
        epsilon_val = float(self.p1_epsilon.text())
        min_samples_val = float(self.p1_min_samples.text())
        density_th_val2 = float(self.p1_density_threshold2.text())
        density_th_val3 = float(self.p1_density_threshold3.text())
        pc_filter_val = float(self.p1_photoncount.text())
        stdev_num_val = float(self.p1_stdev_num.text())
        ### CALL DBSCAN

    def run_HDBSCAN(self):
        min_pts_val = float(self.p2_min_pts.text())
        epsilon = float(self.p2_epsilon.text())
        min_samples = float(self.p2_min_samples.text())
        extracting_alg = self.p2_extracting_alg.text()
        selection_alpha = float(self.p2_selection_alpha.text())
        density_th_val2 = float(self.p2_density_threshold2.text())
        density_th_val3 = float(self.p2_density_threshold3.text())
        pc_filter_val = float(self.p2_photoncount.text())
        stdev_num_val = float(self.p2_stdev_num.text())
        ### CALL HDBSCAN


    def run_FOCAL(self):
        grid_size = float(self.p3_sigma.text())
        minL = float(self.p3_minL.text())
        minC = float(self.p3_minC.text())
        density_th_val2 = float(self.p3_density_threshold2.text())
        density_th_val3 = float(self.p3_density_threshold3.text())
        pc_filter_val = float(self.p3_photoncount.text())
        stdev_num_val = float(self.p3_stdev_num.text())
        ### CALL FOCAL
        

    def close_window(self):
        self.param_box.close()
        self.param_box = None



if __name__ == '__main__':
    app = QApplication([])
    win = MainWindow()
    win.show()

    app.exec_() #Kickstart the Qt event loop



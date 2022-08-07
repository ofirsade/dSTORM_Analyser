
from PyQt5.QtWidgets import *
"""QApplication, QMainWindow, QLabel, QGridLayout, \
  QWidget, QPushButton, QTableWidget, QTableWidgetItem, QInputDialog, \
  QMessageBox, QComboBox, QLineEdit, QAction, QCheckBox, QFileDialog, QVBoxLayout"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import * #QIcon
from PyQt5.QtCore import * #pyqtSlot
import sys
from os import listdir, path, walk, mkdir, unlink, environ
from shutil import rmtree, copyfile
from datetime import datetime


class AlignDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.init_UI()
        self.DBSCAN = False
        self.HDBSCAN = False
        self.FOCAL = False
    
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
        
        
    def btnstate(self, cb):
        self.num_of_confs = 0
        if cb.text() == 'DBSCAN':
            if cb.isChecked() == True:
                rn = 6 # Number of rows for widget
                self.p1_min_pts = QLineEdit('0', self)
                self.p1_epsilon = QLineEdit('70', self)
                self.p1_min_samples = QLineEdit('22', self)
                self.p1_density_threshold2 = QLineEdit('0', self)
                self.p1_density_threshold3 = QLineEdit('0', self)
                self.p1_photoncount = QLineEdit('1000', self)
                
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
                    self.p1_layout.addRow("PCA Stdev",self.p1_stdev_num)
                    rn = 7
                
                self.p1.setLayout(self.p1_layout)
                self.overall_layout.addWidget(self.p1, 4, 0, rn, 1)
                self.DBSCAN = True
                self.num_of_confs += 1
            else:
                self.p1.setVisible(False)
                self.DBSCAN = False
                self.num_of_confs -= 1
                
        elif cb.text() == 'HDBSCAN':
            if cb.isChecked() == True:
                rn = 6
                self.p2_min_pts = QLineEdit('50', self)
                self.p2_epsilon = QLineEdit('-9999', self)
                self.p2_min_samples = QLineEdit('1', self)
                self.p2_extracting_alg = QLineEdit('leaf', self)
                self.p2_selection_alpha = QLineEdit('1.0', self)
                self.p2_density_threshold2 = QLineEdit('0', self)
                self.p2_density_threshold3 = QLineEdit('0', self)
                self.p2_photoncount = QLineEdit('1000', self)
                
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
                    self.p2_layout.addRow("PCA Stdev",self.p2_stdev_num)
                    rn = 7
                
                self.p2.setLayout(self.p2_layout)
                self.overall_layout.addWidget(self.p2, 4, 1, rn, 1)
                self.HDBSCAN = True
                self.num_of_confs += 1

            else:
                self.p2.setVisible(False)
                self.HDBSCAN = False
                self.num_of_confs -= 1

        elif cb.text() == 'FOCAL':
            if cb.isChecked() == True:
                rn = 6
                self.p3_sigma = QLineEdit('55', self)
                self.p3_minL = QLineEdit('1', self)
                self.p3_minC = QLineEdit('25', self)
                self.p3_density_threshold2 = QLineEdit('0', self)
                self.p3_density_threshold3 = QLineEdit('0', self)
                self.p3_photoncount = QLineEdit('1000', self)                    

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
                    self.p3_layout.addRow("PCA Stdev",self.p3_stdev_num)
                    rn = 7
                
                self.p3.setLayout(self.p3_layout)
                self.overall_layout.addWidget(self.p3, 4, 2, rn, 1)
                self.FOCAL = True
                self.num_of_confs += 1

            else:
                self.p3.setVisible(False)
                self.FOCAL = False
                self.num_of_confs -= 1

        if cb.isChecked() == True:
            self.btn5 = QPushButton(self)
            self.btn5.setText('Run Scan')
            self.btn5.adjustSize()
            self.btn5.clicked.connect(self.run_scan)
            self.overall_layout.addWidget(self.btn5, 8, 3)

    def run_scan(self):

        pass#for i in range(self.num_of_confs):
            

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
        

    def get_files(self):
        """
        Opens multiple files to scan
        """
        self.fnames = QFileDialog.getOpenFileNames(
            parent = self,
            caption = 'Open files',
            filter = 'Loc files (*.csv *.xlsx)'
            )

    def get_dir(self):
        """
        Opens all files in a directory
        """
        dr = QFileDialog.getExistingDirectory(
            parent = self,
            caption = 'Select a Folder'
            )

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

    """
    #Set the screen widget as the one to display
    window.setCentralWidget(w)
    window.setWindowTitle('dSTORM Analyser')
    window.setMinimumSize(500, 200)
    window.show()
    """

    app.exec_() #Kickstart the Qt event loop



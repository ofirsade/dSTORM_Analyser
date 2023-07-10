import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ast
import scipy
import time
import os
import matplotlib.pyplot as plt
import sys
import csv
import collections
import os
from datetime import datetime
from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem
import pyqtgraph as pg

from PyQt5.QtWidgets import *
##import plotly.offline
##from PyQt5.QtCore import QUrl
##from PyQt5.QtWebEngineWidgets import QWebEngineView


sys.setrecursionlimit(10**6)


##def show_in_window(fig, name):
##    
##    plotly.offline.plot(fig, filename = name, auto_open = True)
##    
##    app = QApplication(sys.argv)
##    web = QWebEngineView()
##    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), name))
##    web.load(QUrl.fromLocalFile(file_path))
##    web.show()
####    sys.exit(app.exec_())


def plot_res(xyzl, cpd, fname, alg, output_path, dt_string, show_plts):
    """
    @params:
    @@xyzl -
    @@cpd - [loc_num, volume, radius, density_2d, density_3d]
    @@fname -
    @@alg -
    @@output_path -
    """

    fig = make_subplots(
        rows = 1, 
        cols = 1, 
##        vertical_spacing=0.05,
##        subplot_titles = fname + alg
    )

    bg_df = xyzl.loc[xyzl['Label'] == -1]
    clustered_df = xyzl.loc[xyzl['Label'] != -1]
    unique_labels = set(clustered_df['Label'].values.tolist())
    # Probe 0
    fig.add_trace(
        go.Scattergl(
            x = bg_df['x'].values,
            y = bg_df['y'].values,
            mode = 'markers',
            marker=dict(
                color='grey',           # set color to an array/list of desired values
                opacity=0.1
            )
        ),
        row = 1, col = 1
    ) 
        
    # Draw clusters
    for i in unique_labels:
        pc = clustered_df.loc[clustered_df['Label'] == i]
        tmp_cpd = cpd[str(i)]
        
        fig.add_trace(
            go.Scattergl(
                x = pc['x'].values,
                y = pc['y'].values,
                mode = 'markers',
                marker = dict(
                    color = i, #cmap(np.sqrt(colocalization[i])),    
##                    colorscale = 'rainbow',
                    colorscale = px.colors.qualitative.Pastel2,
                    opacity = 0.5
                ),
                hovertemplate = ("<b>{text}</b><br><br>".format(text = 'Cluster #' + str(i))
                                 + "Loc No.: {a}<br>".format(a = tmp_cpd[0])
                                 + "Volume: {b}<br>".format(b = tmp_cpd[1])
                                 + "Radius: {c}<br>".format(c = tmp_cpd[2])
                                 + "Density (2D): {d}<br>".format(d = tmp_cpd[3])
                                 + "Density (3D): {e}".format(e = tmp_cpd[4])
                                 + "<extra></extra>")
            ),
            row = 1, col = 1
        )
    
    fig.update_xaxes(range = [0, 19000], row = 1, col = 1)     
    fig.update_yaxes(scaleanchor = 'x', scaleratio = 1, row = 1, col = 1)
    fig.update_layout({'plot_bgcolor': 'whitesmoke'})
    fig.update_layout(showlegend = False)
    fig.update_layout(title = alg + ' ' + fname)
    fig.update_layout(hoverlabel = dict(font_size = 18,
                                        font_color = 'white',
                                        font_family="Rockwell"))


    html_name = dt_string + ' ' + fname + ' ' + alg + '.html'
    html_path = os.path.join(output_path, html_name)
    fig.write_html(html_path)
    if show_plts:
        fig.show()
##    show_in_window(fig, html_name)
    

'''
def plot_3d_res(new_lst, name, minL, minC, sig):
    """
    Creates a 3D plot of a clustering algorithm's results
    Params:
     ** new_lst: labeled localizations to be included in the plot (type list). 
     ** name: the name of the dataset.
    """
    config = {
        'toImageButtonOptions': {
            'format': 'jpeg', # one of png, svg, jpeg, webp
            'filename': name,
            'height': None,
            'width': None,
            'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
    ttl = (name)
    sortd = sorted(new_lst, key = lambda x: x[2])
    for point in sortd:
        point[0] = point[0]
        point[1] = point[1]
        point[2] = point[2]
    fig = px.scatter_3d(sortd, x = [point[0] for point in new_lst],
                        y = [point[1] for point in new_lst],
                        z = [point[2] for point in new_lst],
                        color = [label[3] for label in new_lst],
                        opacity = 0.6, size_max = 0.0001,
                        range_x = [0, 15000], range_y = [2000, 18000],
                        range_x = [0, 2500], range_y = [0, 3000],
                        range_z = [0, 1000],
                        range_z = [-6000, 6000],
                        color_continuous_scale = 'phase',
                        title = 'FOCAL ' + ttl)
                        
    fig.update_layout(scene = dict(
        xaxis = dict(
            backgroundcolor = 'whitesmoke',
            gridcolor = 'white',
            showbackground = True,
            zerolinecolor = 'black',
            tickmode = 'linear',
            tick0 = 0,
            dtick = 200),
        yaxis = dict(
            backgroundcolor = 'gainsboro',
            gridcolor = 'white',
            showbackground = True,
            zerolinecolor = 'black',
            tickmode = 'linear',
            tick0 = 0,
            dtick = 200),
        zaxis = dict(
            backgroundcolor = 'lightgrey',
            gridcolor = 'white',
            showbackground = True,
            zerolinecolor = 'black',
            tickmode = 'linear',
            tick0 = 0,
            dtick = 200)))
##    fig.show(config = config)
    fig.write_html(path + name + '_' + label +  ' ' + y_val + '.html')

def plot_2D_image(pts_df, file_name):

    config = {
          'toImageButtonOptions': {
            'format': 'jpeg', # one of png, svg, jpeg, webp
            'height': 3000,
            'width': 4000,
            'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
          }
        }
    fig = px.scatter(pts_df, x = 'x', y = 'y',
                        opacity = 0.3, size_max = 0.005,
                        range_x = [0, 19000], range_y = [0, 15000], color_continuous_scale = 'rainbow',
                        title = file_name)
    fig.update_layout(legend = dict(
        yanchor = "top",
        y = 0.99,
        xanchor = "left",
        x = 0.01
        ))
    fig.update_layout({'plot_bgcolor': 'whitesmoke'})
##    fig.show(config=config)
    fig.write_html(path + name + '_' + label +  ' ' + y_val + '.html')





def plot_3D_image(pts_df, file_name):
    config = {
          'toImageButtonOptions': {
            'format': 'jpeg', # one of png, svg, jpeg, webp
            'height': 3000,
            'width': 4000,
            'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
          }
        }
    fig = px.scatter_3d(pts_df, x = 'x', y = 'y', z = 'z',
                        opacity = 0.3, size_max = 0.005,
                        range_x = [0, 19000], range_y = [0, 19000],
                        range_z = [-6000, 6000], color_continuous_scale = 'rainbow',
                        title = file_name)
    fig.update_layout(legend = dict(
        yanchor = "top",
        y = 0.99,
        xanchor = "left",
        x = 0.01
        ))
    fig.update_layout({'plot_bgcolor': 'whitesmoke'})
##    fig.show(config=config)
    fig.write_html(path + name + '_' + label +  ' ' + y_val + '.html')
    


def plot_image2(self, pts_lst, file_name):
    ttl = (file_name + " With minL = " + str(self.minL) +
       " minC = " + str(self.minC) + " Sigma = " + str(self.sig) + " Cluster Number = " + str(self.cluster_num))
    config = {
      'toImageButtonOptions': {
        'format': 'jpeg', # one of png, svg, jpeg, webp
        'filename': ttl,
        'height': 3000,
        'width': 4000,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
      }
    }
    
    df = pd.DataFrame(pts_lst, columns = ['x', 'y', 'z', 'Label'])
    
    df['Label'] = df['Label'].astype(str)
    fig = px.scatter(df, x = 'x', y = 'y', color = 'Label',
                     #opacity = 0.5,
                     size_max = 0.001,
                     range_x = [0, 15000], range_y = [2000, 18000],
                     color_discrete_sequence=px.colors.qualitative.Pastel[::-1], #Bold[::-1],
                     title = alg + ' ' + ttl)
    fig.update_layout({'plot_bgcolor': 'whitesmoke'})
    """fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = self.sig
        ),
        yaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = self.sig
            )
    )"""

##    fig.show(config=config)
    fig.write_html(path + name + '_' + label +  ' ' + y_val + '.html')
        
                  
def plot_image(self, pts_lst, file_name):
    ttl = (file_name + " With minL = " + str(self.minL) +
       " minC = " + str(self.minC) + " Sigma = " + str(self.sig) + " Cluster Number = " + str(self.cluster_num))
    config = {
      'toImageButtonOptions': {
        'format': 'jpeg', # one of png, svg, jpeg, webp
        'filename': ttl,
        'height': 3000,
        'width': 4000,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
      }
    }
    fig = go.Figure(data=go.Scatter(
            x=[point[0] for point in pts_lst],
            y=[point[1] for point in pts_lst],
            #marker_size = [label[3] for label in pts_lst],
            mode='markers',
            marker=dict(
                   #color='rgb(255, 178, 102)',
                   color = [label[3] for label in pts_lst],
                   size=10,
                   line=dict(
                        color='DarkSlateGrey',
                               width=1
                   )
            )
    ))
    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = self.sig
        ),
        yaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = self.sig
            )
    )
    fig.show()
    
    fig.update_layout(
        title='FOCAL ' + file_name,
        xaxis_title='X',
        yaxis_title='Y',
        plot_bgcolor = 'white',
        paper_bgcolor = 'whitesmoke',
        font=dict(
             family='Verdana',
             size=16,
             color='black'
        )
    )
    fig.write_html(path + name + '_' + label +  ' ' + y_val + '.html')
'''

'''
def plot_2D_res(pts_df, name, alg, output_path):

    clustered_df = pts_df.loc[pts_df['Label'] != -1]
    background_df = pts_df.loc[pts_df['Label'] == -1]
    fig = make_subplots(rows = 1, cols = 1)
    fig.add_trace(go.Scatter(x = clustered_df['x'], y = clustered_df['y'], mode = 'markers', marker_color = clustered_df['Label']),
                  row = 1, col = 1)
##                         name = 'clusters',
##                         marker_color = 'blue'), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = background_df['x'], y = background_df['y'], mode = 'markers', opacity = 0.2,
                             marker = dict(color = 'lightgrey')),
                             row = 1, col = 1)

    fig.update_layout({'plot_bgcolor': 'whitesmoke'})
    fig.update_layout(autosize = True, title_text = name)
##    fig.write_html(output_path + '/' + name + ' ' + alg + '.html')
    return fig


def plt_main(xyzl, fname, alg, output_path):
    
    fig_2d = plot_2D_res(xyzl, fname, alg, output_path)
    
    now = datetime.now() # datetime object containing current date and time
    dt_string = now.strftime("%Y.%m.%d %H:%M:%S") # YY.mm.dd H:M:S
    path = os.path.join(output_path, "htmls")
    fig_2d.write_html(path + dt_string + ' ' + fname + '_' + alg + '.html')


def plot_main(xyzl, fname, unique_labels, alg, output_path):

    fig = make_subplots(
        rows = 1, 
        cols = 1, 
##        vertical_spacing=0.05,
        subplot_titles=orig_df.filename.to_list()
    )

    
    bg_df = xyzl.loc[xyzl['Label'] == -1]
    clustered_df = xyzl.loc[xyzl['Label'] != -1]
    # Probe 0
    fig.add_trace(
        go.Scattergl(
            x = bg_df['x'].values,
            y = bg_df['y'].values,
            mode = 'markers',
            marker=dict(
                color='grey',           # set color to an array/list of desired values
                opacity=0.1
            )
        ),
        row = 1, col = 1
    )
        
    # Draw clusters
    for i in unique_labels:
        pc = clustered_df.loc[clustered_df['Label'] == i]
        
        fig.add_trace(
            go.Scattergl(
                x = pc['x'].values,
                y = pc['y'].values,
                mode = 'markers',
                marker = dict(
                    color = i, #cmap(np.sqrt(colocalization[i])),    
                    colorscale = 'rainbow',
                    opacity = 0.5
                )
            ),
            row = 1, col = 1
        )
    
    fig.update_xaxes(range = [0, 18e3], row = 1, col = 1)     
    fig.update_yaxes(scaleanchor = 'x', scaleratio = 1, row = 1, col = 1)

    fig.show()
'''    

##class PlotWindow(QWidget):
##
##    def __init__(self):
##        super().__init__()
##        self.init_UI()
##        
##
##    def init_UI(self):
##
####        layout = QVBoxLayout()
##        
##        # Define a top-level widget to hold everything
##        pw = QtWidgets.QWidget()
##        w.setWindowTitle('Scan Results')
##        self.setGeometry(50,50,1600,800)
##        layout_widget = QtGui.QWidget(self)
##        self.setCentralWidget(layout_widget)
##
##        self.grid = QtGui.QGridLayout()
##        layout_widget.setLayout(self.grid)
##
##        self.graph_fr = gl.GLViewWidget()
##        self.grid.addWidget(self.graph_fr,0,0)
##
##        try:
##            self.grid.removeWidget(self.plt_widget)
##            self.plt_widget.setParent(None)
##        except:
##            pass
##        self.graph_fr = gl.GLViewWidget()
##        self.grid.addWidget(self.graph_fr,0,0)
##
##        self.plot()
##        
##        ## Display the widget as a new window
##        w.show()
##
##        ## Start the Qt event loop
##        vis.exec()  # or app.exec_() for PyQt5 / PySide2
##
##
##    def plot():
##
##        
##        
##
##    def show_points(self):
##        '''
##        Adds the localisations to the display in the main window
##        '''
##        self.plt_table = self.xyzl.copy() #Makes a deep copy of self.loc_table
##
##        x_coords = self.xyzl['x']
##        y_coords = self.xyzl['y']
##        z_coords = self.xyzl['z']
##         
##        self.mid_x = np.median(x_coords)
##        self.mid_y = np.median(y_coords)
##        self.mid_z = np.median(z_coords)
##        
##        mid_coords = np.ptp(self.plt_table,axis=0) + np.amin(self.plt_table,axis=0)
##        
##        self.plt_table -= mid_coords
##        
##        self.sp = gl.GLScatterPlotItem(pos = self.plt_table,
##                                       color = (1.0,1.0,1.0,1.0),
##                                       size = 1.0,pxMode=False)
##        
##        self.sp.translate(self.mid_x,self.mid_y,self.mid_z)
###            self.sp.rotate(90, 1, 0, 0)
###            self.sp.rotate(90, 0, 1, 0)
###            self.sp.translate(-mid_x, -mid_)
##        self.graph_fr.addItem(self.sp)
##        self.graph_fr.update()

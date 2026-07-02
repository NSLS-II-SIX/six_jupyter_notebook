from os import (getcwd, path)
cwd = getcwd()
import numpy as np
from time import (strftime)

from h5py import File as h5_file
from glob import glob as globf
# from itertools import cycle
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mpl_color
from ipywidgets import interactive_output as ipy_interact
from ipywidgets import (HBox, VBox, FloatSlider, IntSlider, SelectionSlider,Layout,Button,ButtonStyle)
from ipywidgets.widgets import Dropdown
import ipywidgets as widgets

from IPython.display import display
from Loc_Funcs import *


""
def m_colormap(colors, color_bin=10):
    c_array = list([])
    for i in colors:
        c_array.append(mpl_color.to_rgb(i))
    color_map = mpl_color.LinearSegmentedColormap.from_list('m_color', c_array, N=color_bin)

    return color_map(np.linspace(0, 1, color_bin))


""
def img_plot(x, y, border_line=1500, fig=None, plt_close=1):
    """Display the RIXS data as an image although x and y are one-dimensional!"""

    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax = fig.add_subplot(111)
        pos1 = ax.get_position()  # pos=[left, bottom, width, height]
        ax.set_position([pos1.x0 + 0.0, pos1.y0 - 0.005, pos1.width + 0.08, pos1.height + 0.1])  # For center
    else:
        ax = plt.gca()
        plt.cla()

    # Plot the data
    ax.plot(x[x <= border_line], y[x <= border_line],
            linestyle='none', marker='o', markersize=2, mfc='royalblue', mec='royalblue')

    ax.plot(x[x > border_line], y[x > border_line],
            linestyle='none', marker='o', markersize=2, mfc='deeppink', mec='deeppink')

    ax.axvline(x=border_line, linestyle='-', color='k')

    ax.text(0.07, 0.04, 'L',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'},
            color='royalblue', fontdict={'size': 20, 'weight': 'bold', 'name': 'Helvetica'})

    ax.text(0.98, 0.04, 'R',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'},
            color='deeppink', fontdict={'size': 20, 'weight': 'bold', 'name': 'Helvetica'})

    ax.set_xlabel('x/Pixel', fontdict={'size': 15, 'name': 'Helvetica'})
    ax.set_ylabel('y/Pixel', fontdict={'size': 15, 'name': 'Helvetica'})
    ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
    plt.show()

    return ax


""
def spec_plot(data, scan=None, sig='spec', xshift=0, yshift=0, fig=None, plt_close=1):
    ##############################################################
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))

    ##############################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
        fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharex=ax1)
    else:
        fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        plt.cla()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharex=ax2)
    ##############################################################
    colors = m_colormap(['royalblue', 'deeppink'], color_bin=len(scan))
    sensor = ['l', 'r']

    def sig_plot(x, y, ax, i, color):
        # nonlocal xshift
        # nonlocal yshift

        # ax.clear()

        ax.plot(x + xshift * i, y + yshift * i, linestyle='-', marker='.', color=color)
        ax.set_xlabel('pixel', fontdict={'size': 15})
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

    x_trial = data['six-' + str(scan[0])][sig + '_x_l']

    def view_sig(x_low=x_trial.min(), x_high=x_trial.max()):

        # nonlocal data
        # nonlocal scan
        # nonlocal sensor

        for k, m in enumerate(sensor):
            for i, n in enumerate(scan):
                x = data['six-' + str(n)][sig + '_x_' + m]
                y = data['six-' + str(n)][sig + '_y_' + m]
                sig_plot(x, y, ax=eval('ax' + str(k + 1)), i=i, color=colors[i, :])

        ax1.set_title('Left Sensor', fontdict={'size': 16})
        ax2.set_title('Right Sensor', fontdict={'size': 16})

        if x_high <= x_low:
            x_high = x_low + 10
        ax1.set_xlim(x_low, x_high)
        ax1.set_ylabel('photon No.', fontdict={'size': 15})

    mwidget = ipy_interact(view_sig, x_low=(x_trial.min(), x_trial.max(), 5), x_high=(x_trial.min(), x_trial.max(), 5))

    display(mwidget)

    return mwidget

    


""
def scan_plot(scan, sample=None, meta=None, fig=None, plt_close=1):
    if sample is None:
        sample = 'six'
    #########################################################
    scan = np.sort(scan)
    data = {}
    det_list_start = []
    for i,n in enumerate(scan):
        det_list = []
        print('It is loading the scan: six-{}'.format(n))
        data['six-' + str(n)] = scan_data(n, meta=meta)
        if i==0:
            for mkey in data['six-' + str(n)]['data'].keys():
                if mkey[:4] in ['sclr','rixs','ring','k263']:
                    det_list_start.append(mkey)
        else:
            for mkey in data['six-' + str(n)]['data'].keys():
                if mkey[:4] in ['sclr','rixs','ring','k263']:
                    det_list.append(mkey) 
            det_list_start = list(set.intersection(*map(set, [det_list_start,det_list])))
            
                
    #########################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Single Scan Plot', figsize=(10, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharex=ax1)
    else:
        fig = plt.gcf()
        fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        plt.cla()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharex=ax1)
    ##############################################################
    color_seed = ['royalblue', 'deeppink']
    if len(scan) == 1:
        colors = m_colormap(color_seed, color_bin=2)
    else:
        colors = m_colormap(color_seed, color_bin=len(scan) + 1)

    #########################################################
    
    def sig_plot(x, y, ax, color, label, xlim=None):
        ax.plot(x, y, linestyle='-', marker='.', color=color, label=label)

    def norm_plot(x,y, ax, color, label):
        
        if len(x)!=len(y):
            x_new = x[:int(min(len(x),len(y)))]
            y_new = y[:int(min(len(x),len(y)))]
        else:
            x_new,y_new = x,y
        
        ax.plot(x_new, y_new/y_new[0], linestyle='--', marker='.', color=color, label=label)

    ############################################################################################################################
    def view_sig_k2636b(i=0, detector = det_list_start[0],
                        save='No', disp_type='single'):
        ax1.clear()
        ax2.clear()
        ##############################################################
        # check the motor size to determine the plot type
        motor = list(data['six-' + str(scan[i])]['motor'].keys())
        motor_size = len(motor)
        print('Loaded Keithley measurement')
        xlim_low, xlim_high = np.array([]), np.array([])

        if disp_type == 'single':
            if motor_size==1:
                motor_local = motor
                x_original = data['six-' + str(scan[i])]['motor'][motor_local[0]]
                y_original = data['six-' + str(scan[i])]['data'][detector]
    
                if len(x_original)!=len(y_original):
                    x = x_original[:int(min(len(x_original),len(y_original)))]
                    y = y_original[:int(min(len(x_original),len(y_original)))]
                else:
                    x,y = x_original,y_original
    
                xlim_low = np.append(xlim_low, x.min())
                xlim_high = np.append(xlim_high, x.max())
    
                sig_plot(x, y, ax=ax1, color=colors[i, :], label='six-' + str(scan[i]))

                sig_plot(x, y, ax=ax2, color=colors[i, :], label='six-' + str(scan[i]))
    
                # ax1.set_xlim(xlim_low.min()-1,xlim_high.max()+1)
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax1.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                ax1.set_xlabel(motor_local[0], fontdict={'size': 15})
                ax1.set_ylabel(f'{detector}/arbi.', fontdict={'size': 15})
                ax1.legend()
    
                ax2.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                ax2.set_xlabel(motor_local[0], fontdict={'size': 15})
                ax2.set_ylabel(f'{detector}/arbi.', fontdict={'size': 15})
                ax2.legend()
    
                # print(data['six-' + str(scan[i])].keys())
                # print(data['six-' + str(scan[i])]['data'].keys())
                # print(data['six-' + str(scan[i])]['meta'].keys())
            else:
                print('Unable to handle the data collected by moving multiple (>2) motors!!!')
        else:
            if motor_size==1:
                for n in range(0, i + 1, 1):   
                    motor_local = motor
                    x_original = data['six-' + str(scan[n])]['motor'][motor_local[0]]
                    y_original = data['six-' + str(scan[n])]['data'][detector]

                    if len(x_original)!=len(y_original):
                        x = x_original[:int(min(len(x_original),len(y_original)))]
                        y = y_original[:int(min(len(x_original),len(y_original)))]
                    else:
                        x,y = x_original,y_original

                    xlim_low = np.append(xlim_low, x.min())
                    xlim_high = np.append(xlim_high, x.max())

                    sig_plot(x, y, ax=ax1, color=colors[n, :], label='six-' + str(scan[n]))
                    
                    sig_plot(x, y, ax=ax2, color=colors[n, :], label='six-' + str(scan[n]))
                
                # ax1.set_xlim(xlim_low.min()-1,xlim_high.max()+1)
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax1.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                ax1.set_xlabel(motor_local[0], fontdict={'size': 15})
                ax1.set_ylabel(f'{detector}/arbi.', fontdict={'size': 15})
                ax1.legend()

                ax2.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                ax2.set_xlabel(motor_local[0], fontdict={'size': 15})
                ax2.set_ylabel(f'{detector}/arbi.', fontdict={'size': 15})
                ax2.legend()
            
            else:
                print('Can not display multiple images on one plot!!!')
                
        if save == 'No':
            pass
        elif save == 'hdf':
            save_folder = cwd + '/Data/'
            if disp_type == 'single':
                save_scan(data, save_folder, data_format='hdf', scan=scan[i], sample=sample)
                print('Data has been saved into HDF files!!!')
            else:
                save_scan(data, save_folder, data_format='hdf', sample=sample)
                print('All data has been saved into HDF files!!!')
        else:
            save_folder = cwd + '/Data/'
            if disp_type == 'single':
                save_scan(data, save_folder, data_format='txt', scan=scan[i], sample=sample)
                print('Data has been saved into TXT files!!!')
            else:
                save_scan(data, save_folder, data_format='txt',sample=sample)
                print('All data has been saved into TXT files!!!')
        ##############################################################
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################
    def view_sig(i=0, detector='sclr_channels_chan2',
                 norm='No',norm_chan='sclr_channels_chan8', 
                 save='No', disp_type='single'):
        ax1.clear()
        ax2.clear()
        ##############################################################
        # check the motor size to determine the plot type
        # motor = []
        # for mkey in data['six-' + str(scan[i])]['data'].keys():
        #     if mkey[:4] not in ['sclr','rixs','ring']:
        #         motor.append(mkey)
        motor = list(data['six-' + str(scan[i])]['motor'].keys())        
        motor_size = len(motor)
        ##############################################################
        # Set-up the data normalization
        for t in scan:
            if norm == 'No':
                norm_f = np.ones(np.shape(data['six-' + str(t)]['data'][norm_chan]))
            elif norm == 'Yes':
                norm_f = data['six-' + str(t)]['data'][norm_chan]
            else:
                print('Invalid norm opition, only Yes or No!!!')
            for mkey in det_list_start:
                if mkey[:4] in ['sclr','rixs','ring','k263']:
                    if mkey!=norm_chan:
                        data['six-' + str(t)]['data'][mkey+'_norm'] = data['six-' + str(t)]['data'][mkey]/ (norm_f / np.ravel(norm_f)[0])
                    else:
                        pass
                else:
                    pass
            data['six-' + str(t)]['meta']['norm'] = norm       
            data['six-' + str(t)]['meta']['norm_chan'] = norm_chan
        ##############################################################
        xlim_low, xlim_high = np.array([]), np.array([])

        if disp_type == 'single':
            
            if motor_size==1:
                # motor_local = [p for p in data['six-' + str(scan[i])]['data'].keys() if p[:4]!=['sclr','rixs','ring']]
                motor_local = list(data['six-' + str(scan[i])]['motor'].keys())    
                x_original = data['six-' + str(scan[i])]['motor'][motor_local[0]]
                try:
                    y_original = data['six-' + str(scan[i])]['data'][detector+'_norm']
                except:
                    y_original = data['six-' + str(scan[i])]['data'][detector]

                if len(x_original)!=len(y_original):
                    x = x_original[:int(min(len(x_original),len(y_original)))]
                    y = y_original[:int(min(len(x_original),len(y_original)))]
                else:
                    x,y = x_original,y_original

                xlim_low = np.append(xlim_low, x.min())
                xlim_high = np.append(xlim_high, x.max())

                sig_plot(x, y, ax=ax1, color=colors[i, :], label='six-' + str(scan[i]))

                norm_factor = data['six-' + str(scan[i])]['data'][norm_chan]
                norm_plot(x,norm_factor, ax=ax2, color=colors[i, :], label='six-' + str(scan[i]))

                # ax1.set_xlim(xlim_low.min()-1,xlim_high.max()+1)
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax1.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                ax1.set_xlabel(motor_local[0], fontdict={'size': 15})
                ax1.set_ylabel('Intensity/arbi.', fontdict={'size': 15})
                ax1.legend()

                ax2.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                ax2.set_xlabel(motor_local[0], fontdict={'size': 15})
                ax2.set_ylabel('norm_I0/arbi.', fontdict={'size': 15})
                ax2.legend()

                # print(data['six-' + str(scan[i])].keys())
                # print(data['six-' + str(scan[i])]['data'].keys())
                # print(data['six-' + str(scan[i])]['meta'].keys())
            elif motor_size==2:
                # motor_local = [p for p in data['six-' + str(scan[i])]['data'].keys() if p[:4]!=['sclr','rixs','ring']]
                motor_local = list(data['six-' + str(scan[i])]['motor'].keys())    
                x_original = data['six-' + str(scan[i])]['motor'][motor_local[0]]
                y_original = data['six-' + str(scan[i])]['motor'][motor_local[1]]

                try:
                    z_original = data['six-' + str(scan[i])]['data'][detector+'_norm'] 
                except:
                    z_original = data['six-' + str(scan[i])]['data'][detector] 
                
                if len(np.unique(len(x_original),len(y_original),len(z_original)))!=1:
                    x = x_original[:int(min(len(x_original),len(y_original),len(z_original)))]
                    y = y_original[:int(min(len(x_original),len(y_original),len(z_original)))]
                    z = z_original[:int(min(len(x_original),len(y_original),len(z_original)))]
                else:
                    x,y,z = x_original,y_original,z_original
                norm_factor = data['six-' + str(scan[i])]['data'][norm_chan]
                ##############################################################
                ax1.scatter(x,y,c=z,s=40,cmap='nipy_spectral',marker='s')
                ax2.scatter(x,y,c=norm_factor,s=40,cmap='rainbow',marker='s')
                
                
                ##############################################################
                ax1.set_xlim(x.min() - np.abs(x[3]-x[2]), x.max() + np.abs(x[3]-x[2]))
                ax1.set_ylim(y.min() - np.abs(y[3]-y[2]), y.max() + np.abs(y[3]-y[2]))
                ax1.set_xlabel(motor_local[0], fontdict={'size': 15})
                ax1.set_ylabel(motor_local[1], fontdict={'size': 15})
                ax1.set_title('six-'+str(scan[i]),fontdict={'size': 15})
                
                ax2.set_xlim(x.min() - np.abs(x[3]-x[2]), x.max() + np.abs(x[3]-x[2]))
                ax2.set_ylim(y.min() - np.abs(y[3]-y[2]), y.max() + np.abs(y[3]-y[2]))
                ax2.set_xlabel(motor_local[0], fontdict={'size': 15})
                ax2.set_ylabel(motor_local[1], fontdict={'size': 15})
                ax2.set_title('six-'+str(scan[i]),fontdict={'size': 15})
                ##############################################################
                
            else:
                print('Unable to handle the data collected by moving multiple (>2) motors!!!')
            
        else:
            if motor_size==1:
                motor_local_list = []
                for n in range(0, i + 1, 1):
                    # motor_local = [p for p in data['six-' + str(scan[n])]['data'].keys() if p[:4]!=['sclr','rixs','ring']]
                    motor_local = list(data['six-' + str(scan[i])]['motor'].keys())    
                    motor_local_list.append(motor_local[0])
                
                    x_original = data['six-' + str(scan[n])]['motor'][motor_local[0]]
                    
                    try:
                        y_original = data['six-' + str(scan[n])]['data'][detector+'_norm']
                    except:
                        y_original = data['six-' + str(scan[n])]['data'][detector]

                    if len(x_original)!=len(y_original):
                        x = x_original[:int(min(len(x_original),len(y_original)))]
                        y = y_original[:int(min(len(x_original),len(y_original)))]
                    else:
                        x,y = x_original,y_original

                    xlim_low = np.append(xlim_low, x.min())
                    xlim_high = np.append(xlim_high, x.max())

                    sig_plot(x, y, ax=ax1, color=colors[n, :], label='six-' + str(scan[n]))

                    norm_factor = data['six-' + str(scan[n])]['data'][norm_chan]
                    norm_plot(x,norm_factor, ax=ax2, color=colors[n, :], label='six-' + str(scan[n]))
                
                # ax1.set_xlim(xlim_low.min()-1,xlim_high.max()+1)
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax1.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                if len(set(motor_local_list))==1:
                    ax1.set_xlabel(motor_local[0], fontdict={'size': 15})
                else:
                    ax1.set_xlabel('N/A', fontdict={'size': 15})
                ax1.set_ylabel('Intensity/arbi.', fontdict={'size': 15})
                ax1.legend()

                ax2.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                if len(set(motor_local_list))==1:
                    ax2.set_xlabel(motor_local[0], fontdict={'size': 15})
                else:
                    ax2.set_xlabel('N/A', fontdict={'size': 15})
                ax2.set_ylabel('norm_I0/arbi.', fontdict={'size': 15})
                ax2.legend()
            
            else:
                print('Can not display multiple images on one plot!!!')

        ##############################################################
        if save == 'No':
            pass
        elif save == 'hdf':
            save_folder = cwd + '/Data/'
            if disp_type == 'single':
                save_scan(data, save_folder, data_format='hdf', scan=scan[i], sample=sample)
                print('Data has been saved into HDF files!!!')
            else:
                save_scan(data, save_folder, data_format='hdf', sample=sample)
                print('All data has been saved into HDF files!!!')
        else:
            save_folder = cwd + '/Data/'
            if disp_type == 'single':
                save_scan(data, save_folder, data_format='txt', scan=scan[i], sample=sample)
                print('Data has been saved into TXT files!!!')
            else:
                save_scan(data, save_folder, data_format='txt',sample=sample)
                print('All data has been saved into TXT files!!!')
        ##############################################################
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    ####################################################################################################################
    i_s = IntSlider(min=0, max=len(scan) - 1, step=1, value=0, description='i')
    i_s.style.handle_color = 'black'

    if det_list_start[0][:4]=='k263':
        detector_s = Dropdown(options=[(k,k) for k in det_list_start], description='detector')
    else:
        detector_s = Dropdown(options=[(k,k) for k in det_list_start], value='sclr_channels_chan2', description='detector')

    
    if det_list_start[0][:4]!='k263':
        norm_s = Dropdown(options=[('No', 'No'), ('Yes', 'Yes')], value='No', description='norm')
        norm_chan_s = Dropdown(options=[(k,k) for k in det_list_start if k[:4]!='rixs'], value='sclr_channels_chan8', description='norm_chan')
    else:
        pass
    
    save_s = Dropdown(options=[('Yes_hdf', 'hdf'), ('Yes_txt', 'txt'), ('No', 'No')], value='No', description='save!!!')
    disp_type_s = Dropdown(options=[('single', 'single'), ('multiple', 'multiple')], value='single',
                           description='disp_type')
    if det_list_start[0][:4]!='k263':
        mwidget = ipy_interact(view_sig,
                               {'i': i_s, 'detector':detector_s,
                                'norm': norm_s,'norm_chan': norm_chan_s, 
                                'save': save_s, 'disp_type': disp_type_s})
    else:
        mwidget = ipy_interact(view_sig_k2636b,
                               {'i': i_s, 'detector':detector_s,
                                'save': save_s, 'disp_type': disp_type_s})

    left_box = VBox([detector_s,i_s])
    if det_list_start[0][:4]!='k263':
        center_box = VBox([norm_s,norm_chan_s])
    else:
        pass
    right_box = VBox([disp_type_s, save_s])
    if det_list_start[0][:4]!='k263':
        display(HBox([left_box, center_box,right_box]), mwidget)  # Show all controls
    else:
        display(HBox([left_box,right_box]), mwidget)  # Show all controls
""
def raw_sig(raw_data, *sig_roi, scan=None, fig=None, plt_close=1):
    """Check all data signal and beamline status!"""

    #     data = {**raw_data}
    data = deepcopy(raw_data)

    # specify the border line between left and right sensors!
    border = raw_data['six-' + str(scan[0])]['image_size'][0] / 2

    ###########################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Browse Raw Signal', figsize=(10, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        fig = plt.gcf()
        fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        # plt.cla()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    ##############################################################################################
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in raw_data.keys():
            scan = np.append(scan, int(n[4:]))
    scan_len = len(scan)

    ##############################################################################################
    # Display the raw signal
    def img_plot(x, y, ax=ax1, border_line=border):
        ax.clear()

        # Plot the data
        ax.plot(x[x <= border_line], y[x <= border_line],
                linestyle='none', marker='o', markersize=2, mfc='royalblue', mec='royalblue', alpha=0.8)

        ax.plot(x[x > border_line], y[x > border_line],
                linestyle='none', marker='o', markersize=2, mfc='deeppink', mec='deeppink', alpha=0.8)

        ax.axvline(x=border_line, linestyle='-', color='k')

        ax.text(0.07, 0.04, 'L',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'},
                color='royalblue', fontdict={'size': 20, 'weight': 'bold'})

        ax.text(0.98, 0.04, 'R',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'},
                color='deeppink', fontdict={'size': 20, 'weight': 'bold'})

        ax.set_xlabel('x/pixel', fontdict={'size': 15})
        ax.set_ylabel('y/pixel', fontdict={'size': 15})
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

        ax.set_xlim(-4, 3304)

        ##############################################################################################

    # Extract the ring current and norm_I0 to check the beamline status
    norm_I0, ring_curr, sig_total_l, sig_total_r = np.array([]), np.array([]), np.array([]), np.array([])
    # data_point = np.array([])
    for n in scan:
        norm_I0 = np.append(norm_I0, raw_data['six-' + str(n)]['norm_I0'])
        ring_curr = np.append(ring_curr, raw_data['six-' + str(n)]['ring_curr'])
        sig_total_l = np.append(sig_total_l,
                                len((raw_data['six-' + str(n)]['sig_y'])[
                                        (raw_data['six-' + str(n)]['sig_x']) <= border]))
        sig_total_r = np.append(sig_total_r,
                                len((raw_data['six-' + str(n)]['sig_y'])[
                                        (raw_data['six-' + str(n)]['sig_x']) > border]))
        # data_point = np.append(data_point,len(raw_data['six-' + str(n)]['sig_y'][(raw_data['six-' + str(n)]['sig_x']) > border]))

    # print(sig_total_r)

    norm_I0 /= norm_I0[0]
    ring_curr /= ring_curr[0]
    sig_total_l /= sig_total_l[0]
    sig_total_r /= sig_total_r[0]

    # data_point/=data_point[0]

    def norm_plot(i, scan=scan, ax=ax2, norm_I0=norm_I0, ring_curr=ring_curr, sig_total_l=sig_total_l,
                  sig_total_r=sig_total_r):
        ax.clear()
        ax.plot(scan, sig_total_l, linestyle='none', marker='.', mfc='royalblue', mec='royalblue', markersize=9,
                label='sig_I_L')
        ax.plot(scan, sig_total_r, linestyle='none', marker='.', mfc='deeppink', mec='deeppink', markersize=9,
                label='sig_I_R')
        ax.plot(scan, norm_I0, linestyle='none', marker='o', mfc='w', mec='k', markersize=9, alpha=0.8, label='I0')

        # ax.plot(scan, data_point, linestyle='-', marker='*', mfc='k', mec='k', markersize=9, alpha=0.8, label='data_point')

        ax.plot(scan, ring_curr, linestyle='none', marker='s', mfc='orange', mec='orange', markersize=9, alpha=0.5,
                label='ring_curr')
        ax.set_xlabel('Scan No.', fontdict={'size': 15})
        ax.set_ylabel('Norm Inten (arbi.)', fontdict={'size': 15})
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

        ax.annotate(str(scan[i]), xy=(scan[i], sig_total_l[i]),
                    xytext=(scan[i], sig_total_l[i] + (sig_total_l.max() - sig_total_l.min()) * 0.2),
                    arrowprops=dict(facecolor='b', edgecolor='b', width=0.8, headwidth=5),
                    fontsize=12, color='b')
        ax.legend()
        # handles,labels = ax.get_legend_handles_labels()
        # leg=ax.legend(handles,labels,loc='upper right',
        #          fontsize=14,numpoints=1,markerscale=1.0,frameon=True,fancybox=True,
        #          borderpad=0.4,handlelength=1,handletextpad=0.2,borderaxespad=0.3,
        #          markerfirst=True)
        # leg.get_frame().set_facecolor('w')

    def view_data(i=0):

        # nonlocal border  # set the border as a nonlocal variable!
        # nonlocal scan  # set the scan as a nonlocal variable!

        x = raw_data['six-' + str(scan[i])]['sig_x']
        y = raw_data['six-' + str(scan[i])]['sig_y']
        #########################################################
        img_plot(x, y)

        if (len(sig_roi) == 0) or (len(sig_roi) > 4):
            pass

        else:
            for v_cut in sig_roi:
                ax1.axvline(v_cut, linestyle='--', color='navy', linewidth=2)

        ax1.set_title('Scan No. is {}'.format('six-' + str(scan[i])), fontdict={'size': 16})

        #####################################
        norm_plot(i)
        #####################################
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################

    i_s = IntSlider(min=0, max=scan_len - 1, step=1, value=0, description='i')
    i_s.style.handle_color = 'black'

    mwidget = ipy_interact(view_data, {'i': i_s})

    display(VBox([i_s]), mwidget)

    ###############################################################################################################
    if (len(sig_roi) == 0) or (len(sig_roi) > 4):
        pass
    elif len(sig_roi) == 1:
        for nn in scan:
            x_temp = deepcopy(data['six-' + str(int(nn))]['sig_x'])
            y_temp = deepcopy(data['six-' + str(int(nn))]['sig_y'])

            y_temp = y_temp[x_temp > sig_roi[0]]
            x_temp = x_temp[x_temp > sig_roi[0]]

            data['six-' + str(int(nn))]['sig_y'] = y_temp
            data['six-' + str(int(nn))]['sig_x'] = x_temp

    elif len(sig_roi) == 2:
        for nn in scan:
            y_temp_l = deepcopy(
                data['six-' + str(int(nn))]['sig_y'][data['six-' + str(int(nn))]['sig_x'] <= border])
            x_temp_l = deepcopy(
                data['six-' + str(int(nn))]['sig_x'][data['six-' + str(int(nn))]['sig_x'] <= border])

            y_temp_r = deepcopy(
                data['six-' + str(int(nn))]['sig_y'][data['six-' + str(int(nn))]['sig_x'] > border])
            x_temp_r = deepcopy(
                data['six-' + str(int(nn))]['sig_x'][data['six-' + str(int(nn))]['sig_x'] > border])

            y_temp_l = y_temp_l[(x_temp_l > sig_roi[0]) & (x_temp_l < sig_roi[1])]
            x_temp_l = x_temp_l[(x_temp_l > sig_roi[0]) & (x_temp_l < sig_roi[1])]

            data['six-' + str(int(nn))]['sig_y'] = np.hstack((y_temp_l,y_temp_r))
            data['six-' + str(int(nn))]['sig_x'] = np.hstack((x_temp_l, x_temp_r))

    elif len(sig_roi) == 3:
        for nn in scan:
            y_temp_l = deepcopy(
                data['six-' + str(int(nn))]['sig_y'][data['six-' + str(int(nn))]['sig_x'] <= border])
            x_temp_l = deepcopy(
                data['six-' + str(int(nn))]['sig_x'][data['six-' + str(int(nn))]['sig_x'] <= border])

            y_temp_r = deepcopy(
                data['six-' + str(int(nn))]['sig_y'][data['six-' + str(int(nn))]['sig_x'] > border])
            x_temp_r = deepcopy(
                data['six-' + str(int(nn))]['sig_x'][data['six-' + str(int(nn))]['sig_x'] > border])

            y_temp_l = y_temp_l[(x_temp_l > sig_roi[0]) & (x_temp_l < sig_roi[1])]
            x_temp_l = x_temp_l[(x_temp_l > sig_roi[0]) & (x_temp_l < sig_roi[1])]

            y_temp_r = y_temp_r[x_temp_r > sig_roi[2]]
            x_temp_r = x_temp_r[x_temp_r > sig_roi[2]]

            data['six-' + str(int(nn))]['sig_y'] = np.hstack((y_temp_l,y_temp_r))
            data['six-' + str(int(nn))]['sig_x'] = np.hstack((x_temp_l, x_temp_r))


    else:
        for nn in scan:
            y_temp_l = deepcopy(
                data['six-' + str(int(nn))]['sig_y'][data['six-' + str(int(nn))]['sig_x'] <= border])
            x_temp_l = deepcopy(
                data['six-' + str(int(nn))]['sig_x'][data['six-' + str(int(nn))]['sig_x'] <= border])

            y_temp_r = deepcopy(
                data['six-' + str(int(nn))]['sig_y'][data['six-' + str(int(nn))]['sig_x'] > border])
            x_temp_r = deepcopy(
                data['six-' + str(int(nn))]['sig_x'][data['six-' + str(int(nn))]['sig_x'] > border])

            y_temp_l = y_temp_l[(x_temp_l > sig_roi[0]) & (x_temp_l < sig_roi[1])]
            x_temp_l = x_temp_l[(x_temp_l > sig_roi[0]) & (x_temp_l < sig_roi[1])]

            y_temp_r = y_temp_r[(x_temp_r > sig_roi[2]) & (x_temp_r < sig_roi[3])]
            x_temp_r = x_temp_r[(x_temp_r > sig_roi[2]) & (x_temp_r < sig_roi[3])]

            data['six-' + str(int(nn))]['sig_y'] = np.hstack((y_temp_l,y_temp_r))
            data['six-' + str(int(nn))]['sig_x'] = np.hstack((x_temp_l, x_temp_r))
    ###############################################################################################################

    return data


""
def check_slope(data, scan=None, slope_l=None, slope_r=None, fig=None, plt_close=1):
    ##############################################################
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))
    scan_len = len(scan)

    # specify the border line between left and right sensors!
    border = data['six-' + str(scan[0])]['image_size'][0]/2
    
    if slope_l is None:
        slope_l = 0
    if slope_r is None:
        slope_r = 0
    ##############################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Check Slope for Both of Sensor', figsize=(10, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharey=ax1)
    else:
        # fig = plt.gcf()
        fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        # plt.cla()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharey=ax1)

    ##############################################################################################
    # Display the raw signal
    def img_plot(x, y, ax, border_line=border, slope=None, sensor='L'):

        if slope is None:
            slope = 0

        ax.clear()

        # Plot the data
        if sensor == 'L':
            ax.plot(x[x <= border_line], y[x <= border_line] - (x[x <= border_line] * slope),
                    linestyle='none', marker='o', markersize=2, mfc='royalblue', mec='royalblue',alpha=0.8)
            ax.set_xlim(0, border_line)
        else:
            ax.plot(x[x > border_line], y[x > border_line] - (x[x > border_line] * slope),
                    linestyle='none', marker='o', markersize=2, mfc='deeppink', mec='deeppink',alpha=0.8)
            ax.set_xlim(border_line, )

        ax.set_xlabel('x/Pixel', fontdict={'size': 15})
        ax.set_ylabel('y/Pixel', fontdict={'size': 15})
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

        ##############################################################################################

    y_pixel_max = (data['six-' + str(scan[0])]['sig_y']).max() + 20  # set the up limit of y-axis

    def view_data(i=0, slope_l=slope_l, slope_r=slope_r,
                  y_bot=0, y_top=y_pixel_max + 5):

        # nonlocal border  # set the border as a nonlocal variable!
        # nonlocal scan  # set the scan as a nonlocal variable!

        x = data['six-' + str(scan[i])]['sig_x']
        y = data['six-' + str(scan[i])]['sig_y']
        ##################################################################

        img_plot(x, y, ax=ax1, slope=slope_l, sensor='L', border_line=border)
        img_plot(x, y, ax=ax2, slope=slope_r, sensor='R', border_line=border)

        ax1.set_title('{}--Left Sensor'.format('six-' + str(scan[i])), fontdict={'size': 16})
        ax2.set_title('{}--Right Sensor'.format('six-' + str(scan[i])), fontdict={'size': 16})

        if y_top <= y_bot:
            y_top = y_bot + 10
        ax1.set_ylim(y_bot, y_top)
        # print('The current slope_l is {}'.format(slope_l))
        # print('The current slope_r is {}'.format(slope_r))
        
        #####################################
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################

    i_s = IntSlider(min=0, max=scan_len - 1, step=1, value=0, description='i')

    slope_l_value = np.arange(slope_l - 2 * np.abs(slope_l), slope_l + 2.005 * np.abs(slope_l), 0.01 * np.abs(slope_l))
    slope_r_value = np.arange(slope_r - 2 * np.abs(slope_r), slope_r + 2.005 * np.abs(slope_r), 0.01 * np.abs(slope_r))

    slope_l_s = SelectionSlider(options=[("%g" % t, t) for t in slope_l_value],
                                value=slope_l_value[int(len(slope_l_value) / 2)], description='slope_l')
    slope_r_s = SelectionSlider(options=[("%g" % t, t) for t in slope_r_value],
                                value=slope_r_value[int(len(slope_r_value) / 2)], description='slope_r')

    y_bot_s = FloatSlider(min=0, max=y_pixel_max, step=1, value=0, description='y_bot')
    y_top_s = FloatSlider(min=0, max=y_pixel_max + 5, step=1, value=y_pixel_max + 5, description='y_top')

    i_s.style.handle_color = 'black'
    slope_l_s.style.handle_color = 'royalblue'
    slope_r_s.style.handle_color = 'deeppink'

    mwidget = ipy_interact(view_data, {'i': i_s, 'slope_l': slope_l_s, 'slope_r': slope_r_s,
                                       'y_bot': y_bot_s, 'y_top': y_top_s})

    left_box = VBox([slope_l_s, y_bot_s, i_s])
    right_box = VBox([slope_r_s, y_top_s])
    display(HBox([left_box, right_box]), mwidget)  # Show all controls

    return {'slope_l': slope_l_s, 'slope_r': slope_r_s}


""
def stat_plot(data, scan=None,slope_l=None, slope_r=None, xshift=0, yshift=0, fig=None, plt_close=1):
    ##############################################################
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))
    scan_len = len(scan)
    # specify the border line between left and right sensors!
    # border = data['six-' + str(scan[0])]['image_size'][0]/2
    ##############################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Check the Statistics', figsize=(10, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharex=ax1)
    else:
        fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        # plt.cla()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharex=ax2)
    ##############################################################
    # colors = ['brown', 'blueviolet', 'deeppink', 'deeppink', 'olive', 'darkgreen', 'dodgerblue', 'blue']
    ##############################################################
    # color_seed = ['darkslategrey', 'lightseagreen']
    # colors = m_colormap(color_seed, color_bin=8)
    ##############################################################
    color_seed1 = ['royalblue', 'slategrey'] #lightsteelblue
    color_seed2 = ['deeppink', 'slateblue'] #lightpink
    colors1 = m_colormap(color_seed1, color_bin=8)
    colors2 = m_colormap(color_seed2, color_bin=8)
    ##############################################################
    ##############################################################
    sensor = ['l', 'r']

    def sig_plot(x, y, ax, m, color, label=None):
        # nonlocal xshift
        # nonlocal yshift

        # ax.clear()

        if label is None:
            ax.plot(x, y, linestyle='-', marker='.', color=color)
        else:
            ax.plot(x, y, linestyle='-', marker='.', color=color, label=label)
        ax.set_xlabel('pixel', fontdict={'size': 15})
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

    x_trial = data['six-' + str(scan[0])]['sig_y']

    def view_sig(i=0, x_low=x_trial.min(), x_high=x_trial.max() + 10, xshift=0, yshift=0, points_per_pixel_cor=1):
        nonlocal colors1,colors2
        
        ax1.clear()
        ax2.clear()
        for n in range(points_per_pixel_cor):
            data_new = sig2spec(data, scan=scan, slope_l=slope_l, slope_r=slope_r, points_per_pixel=n + 1)

            for k, m in enumerate(sensor):
                x = data_new['six-' + str(scan[i])]['spec_x_' + m]
                y = data_new['six-' + str(scan[i])]['spec_y_' + m]
                norm_time = data_new['six-' + str(scan[i])]['count_time']

                sig_plot(x + xshift * n, y / norm_time + yshift * n, ax=eval('ax' + str(k + 1)), m=n, color=eval('colors'+str(int(k+1)))[n],
                         label=str(n + 1))

        ax1.set_title('{}--Left Sensor'.format('six-' + str(scan[i])), fontdict={'size': 16})
        ax2.set_title('{}--Right Sensor'.format('six-' + str(scan[i])), fontdict={'size': 16})

        ax1.legend()
        ax2.legend()

        if x_high <= x_low:
            x_high = x_low + 10
        ax1.set_xlim(x_low, x_high)
        ax1.set_ylabel('photon No./s', fontdict={'size': 15})
        
        #####################################
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################

    i_s = IntSlider(min=0, max=scan_len - 1, step=1, value=0, description='i')
    i_s.style.handle_color = 'black'
    x_low_s = FloatSlider(min=x_trial.min() - 5, max=x_trial.max() + 10, step=1, value=x_trial.min(),
                          description='x_low')
    x_high_s = FloatSlider(min=x_trial.min() - 5, max=x_trial.max() + 10, step=1, value=x_trial.max(),
                           description='x_high')

    xshift_s = FloatSlider(min=-x_trial.max() / 2, max=x_trial.max() / 2, step=1, value=0, description='xshift')
    yshift_s = FloatSlider(min=-5, max=5, step=0.01, value=0, description='yshift')

    points_per_pixel_s = IntSlider(min=1, max=8, step=1, value=1,
                                   description='points_per_pixel_cor', style={'description_width': 'initial'})
    points_per_pixel_s.style.handle_color = 'blue'
    mwidget = ipy_interact(view_sig, {'i': i_s, 'x_low': x_low_s, 'x_high': x_high_s,
                                      'xshift': xshift_s, 'yshift': yshift_s,
                                      'points_per_pixel_cor': points_per_pixel_s})

    left_box = VBox([x_low_s, x_high_s])
    cen_box = VBox([points_per_pixel_s, i_s])
    right_box = VBox([xshift_s, yshift_s])
    display(HBox([left_box, cen_box, right_box]), mwidget)  # Show all controls

    return {'points_per_pixel_cor': points_per_pixel_s}


""
def check_spec_shift(raw_data, scan=None, 
             border=1500, slope_l=0, slope_r=0, points_per_pixel=1,
             cor_roi_l=None, cor_roi_r=None, cor_interp=0, shift_type=None,
             fig=None, plt_close=1):
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))
            
    # specify the border line between left and right sensors!
    # border = data['six-' + str(scan[0])]['image_size'][0]/2
    ######################################################
    data = sig2spec(raw_data, scan=scan, slope_l=slope_l, slope_r=slope_r,
                    points_per_pixel=points_per_pixel)
    xshift_dict, E0_pxl = shift_dict(raw_data, scan=scan, 
                                     slope_l=slope_l, slope_r=slope_r, points_per_pixel=points_per_pixel,
                                     cor_roi_l=cor_roi_l, cor_roi_r=cor_roi_r, cor_interp=cor_interp)
    x_trail = data['six-' + str(scan[0])]['sig_y'].max()

    if shift_type is None:
        shift_type='self'
    ##############################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Spectra after correlation shifts', figsize=(10, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharey=ax1)
    else:
        fig = plt.gcf()
        fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        # plt.cla()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharey=ax1)

    ##############################################################
    color_seed1 = ['royalblue', 'slategrey']  # lightsteelblue
    color_seed2 = ['deeppink', 'slateblue']  # lightpink
    if len(scan) == 1:
        colors1 = m_colormap(color_seed1, color_bin=2)
        colors2 = m_colormap(color_seed2, color_bin=2)
    else:
        colors1 = m_colormap(color_seed1, color_bin=len(scan) + 1)
        colors2 = m_colormap(color_seed2, color_bin=len(scan) + 1)
    ##############################################################
    sensor = ['l', 'r']
    ############################################################################################################################
    def view_sig(xlow_l = cor_roi_l[0],xhigh_l=cor_roi_l[1],xlow_r=cor_roi_r[0],xhigh_r=cor_roi_r[1],
                 xshift=0,yshift=0,shift_type=shift_type,legend='off'):

        nonlocal colors1,colors2
        ##############################################################
        ax1.clear()
        ax2.clear()
        ##############################################################
        if xhigh_l <= xlow_l:
            xhigh_l = xlow_l + 10

        if xhigh_r <= xlow_r:
            xhigh_r = xlow_r + 10
        ############################################################################################################################

        for i, m in enumerate(sensor):
            for k, n in enumerate(scan):
                norm_time = data['six-' + str(n)]['count_time']
                x = data['six-' + str(n)]['spec_x_' + m]
                y = data['six-' + str(n)]['spec_y_' + m] / norm_time

                if k == 0:
                    x_ref = x
                    y_ref = y
                    ##############################################################
                    # Set-up the E0_pxl_l and E0_pxl_r!!!
                    if m == 'l':
                        E0_pxl_l = x_ref[y_ref[(x_ref >= xlow_l) & (x_ref <= xhigh_l)].argmax()] + xlow_l
                    else:
                        E0_pxl_r = x_ref[y_ref[(x_ref >= xlow_r) & (x_ref <= xhigh_r)].argmax()] + xlow_r
                    ##############################################################
                ##############################################################
                y_new = np.interp(x_ref, x, y)

                ##############################################################
                if shift_type == 'self':
                    ynew_shift = np.interp(x_ref, x_ref + xshift_dict[m][k], y_new)
                elif shift_type == 'L':
                    ynew_shift = np.interp(x_ref, x_ref + xshift_dict['l'][k], y_new)
                elif shift_type == 'R':
                    ynew_shift = np.interp(x_ref, x_ref + xshift_dict['r'][k], y_new)
                else:
                    ynew_shift = np.interp(x_ref, x_ref + xshift_dict['av'][k], y_new)

                ########################################################################################################
                ax = eval('ax'+str(int(i+1)))
                ax.plot(x_ref+xshift*k, ynew_shift+yshift*k, linestyle='-', color=eval('colors'+str(int(i+1)))[k,:],
                        marker='.',mfc=eval('colors'+str(int(i+1)))[k,:],mec=eval('colors'+str(int(i+1)))[k,:],label='six-'+str(int(n)))
                ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)
                ########################################################################################################
        ##############################################################
        ax1.axvline(x=E0_pxl_l, linestyle='-', color='k', lw=0.8)
        ax2.axvline(x=E0_pxl_r, linestyle='-', color='k', lw=0.8)

        # ax1.set_title('Left sensor', fontdict={'size': 18})
        ax1.set_xlabel('pixel', fontdict={'size': 10})
        ax1.set_ylabel('photon No./s', fontdict={'size': 10})
        ax1.set_xlim(xlow_l, xhigh_l)

        # ax2.set_title('Right sensor', fontdict={'size': 18})
        ax2.set_xlabel('pixel', fontdict={'size': 10})
        ax2.set_xlim(xlow_r, xhigh_r)

        ax1.set_title('Left Sensor', fontdict={'size': 13})
        ax2.set_title('Right Sensor', fontdict={'size': 13})
        
        if legend=='off':
            pass
        else:
            ax1.legend()
            ax2.legend()
        #####################################
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################
    ####################################################################################################################
    xlow_l_s = FloatSlider(min=-5, max=x_trail.max(), step=0.1, value=cor_roi_l[0], description='xlow_l')
    xhigh_l_s = FloatSlider(min=0, max=x_trail.max() + 10, step=0.1, value=cor_roi_l[1], description='xhigh_l')
    xlow_r_s = FloatSlider(min=-5, max=x_trail.max(), step=0.1, value=cor_roi_r[0], description='xlow_r')
    xhigh_r_s = FloatSlider(min=0, max=x_trail.max() + 10, step=0.1, value=cor_roi_r[1], description='xhigh_r')

    xshift_s = FloatSlider(min=-200, max=200, step=0.5, value=0, description='xshift')
    yshift_s = FloatSlider(min=-2, max=10, step=0.01, value=0, description='yshift')

    shift_type_s = Dropdown(options=[('self', 'self'), ('L', 'L'), ('R', 'R'), ('AV', 'AV')], value=shift_type,
                            description='shift_tpye')
    legend_s = Dropdown(options=[('OFF', 'off'), ('ON', 'on')], value='off',description='legend')
    ####################################################################################################################
    xlow_l_s.style.handle_color = 'royalblue'
    xhigh_l_s.style.handle_color = 'royalblue'

    xlow_r_s.style.handle_color = 'deeppink'
    xhigh_r_s.style.handle_color = 'deeppink'

    mwidget = ipy_interact(view_sig, {'xlow_l': xlow_l_s, 'xhigh_l': xhigh_l_s,
                                      'xlow_r': xlow_r_s, 'xhigh_r': xhigh_r_s,
                                      'yshift': yshift_s,'xshift': xshift_s,
                                      'shift_type': shift_type_s,'legend':legend_s})

    left_box = VBox([xlow_l_s, xhigh_l_s,xshift_s])
    center_box = VBox([xlow_r_s, xhigh_r_s,yshift_s])
    right_box = VBox([shift_type_s,legend_s])

    display(HBox([left_box, center_box, right_box]), mwidget)  # Show all controls

    

    

""
def check_cor(data, scan=None,
              slope_l=0, slope_r=0, points_per_pixel=1,
              cor_roi=None,
              fig=None, plt_close=1):
            
    # specify the border line between left and right sensors!
    # border = data['six-' + str(scan[0])]['image_size'][0]/2
    
    data = sig2spec(data, scan=scan, slope_l=slope_l, slope_r=slope_r, points_per_pixel=points_per_pixel)
    ##############################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Check the Correlation Shifts', figsize=(7, 5.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(7, 5.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, sharey=ax1)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

    else:
        # fig = plt.gcf()
        fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        # plt.cla()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, sharey=ax1)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

    ##############################################################
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))
    scan.astype(int)

    ##############################################################
    color_seed1 = ['royalblue', 'slategrey'] #lightsteelblue
    color_seed2 = ['deeppink', 'slateblue'] #lightpink
    if len(scan) == 1:
        colors1 = m_colormap(color_seed1, color_bin=2)
        colors2 = m_colormap(color_seed2, color_bin=2)
    else:
        colors1 = m_colormap(color_seed1, color_bin=len(scan) + 1)
        colors2 = m_colormap(color_seed2, color_bin=len(scan) + 1)
    ##############################################################
    sensor = ['l', 'r']

    def sig_plot(x, y, ax, i, color):
        # ax.clear()
        ax.plot(x, y, linestyle='-', color=color)
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

    def xshift_plot(x, y, ax, color, sensor=None, label=None):
        if sensor is None:
            ax.plot(x, y, linestyle='-', marker='.', color=color, label=label)
        else:
            if sensor == 'l':
                ax.plot(x, y, linestyle='none', marker='.', mfc=color, mec=color)
            else:
                ax.plot(x, y, linestyle='--', color=color)

        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

    x_trail_l = data['six-' + str(scan[0])]['sig_y'].max()
    x_trail_r = data['six-' + str(scan[0])]['sig_y'].max()
    
    if cor_roi is None:
        xlow_l_ = -5
        xhigh_l_ = x_trail_l
        xlow_r_ = -5
        xhigh_r_ = x_trail_r
    else:
        xlow_l_ = cor_roi[0][0]
        xhigh_l_ = cor_roi[0][1]
        xlow_r_ = cor_roi[1][0]
        xhigh_r_ = cor_roi[1][1]
    
    def view_sig(xlow_l=xlow_l_, xhigh_l=xhigh_l_,
                 xlow_r=xlow_r_, xhigh_r=xhigh_r_,
                 cor_interp=0, shift_type='self'):

        nonlocal colors1,color_seed1
        nonlocal colors2,color_seed2

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        if xhigh_l <= xlow_l:
            xhigh_l = xlow_l + 10

        if xhigh_r <= xlow_r:
            xhigh_r = xlow_r + 10

        xshift_dict = {}

        for i, m in enumerate(sensor):
            xshift_dict[m] = np.array([])

            cor_roi = np.array([eval('xlow_' + m), eval('xhigh_' + m)])

            for k, n in enumerate(scan):
                x = data['six-' + str(n)]['spec_x_' + m]
                y = data['six-' + str(n)]['spec_y_' + m]

                if k == 0:
                    x_ref = x
                    y_ref = y
                    ##############################################################
                    # Set-up the E0_pxl_l and E0_pxl_r!!!
                    if m == 'l':
                        E0_pxl_l = x_ref[y_ref[(x_ref >= cor_roi[0]) & (x_ref <= cor_roi[1])].argmax()] + cor_roi[0]
                    else:
                        E0_pxl_r = x_ref[y_ref[(x_ref >= cor_roi[0]) & (x_ref <= cor_roi[1])].argmax()] + cor_roi[0]
                    ##############################################################

                x_new = x_ref
                y_new = np.interp(x_ref, x, y)

                xshift_array = np.array([])
                for cor_num in np.arange(0, cor_interp + 1, 1):
                    if cor_num == 0:
                        xshift = 0
                    else:
                        xshift = sig_cor(x_ref=x_ref, y_ref=y_ref, x=x_new, y=y_new, roi_cor=cor_roi,
                                         cor_interp=int(cor_num))
                    xshift_array = np.append(xshift_array, xshift)
                ##############################################################
                xshift_plot(np.arange(0, cor_interp + 1, 1), xshift_array, sensor=m, ax=ax3,
                            color=eval('colors'+str(int(i+1)))[k, :], label=m)

                xshift_dict[m] = np.append(xshift_dict[m], xshift)

            xshift_plot(scan, xshift_dict[m], ax=ax4, color=eval('color_seed'+str(int(i+1)))[0], label=m.upper())

        xshift_dict['av'] = (xshift_dict['l'] + xshift_dict['r']) / 2
        xshift_plot(scan, xshift_dict['av'], ax=ax4, color='seagreen', label='AV')
        ##############################################################
        for i, m in enumerate(sensor):
            for k, n in enumerate(scan):
                x = data['six-' + str(n)]['spec_x_' + m]
                y = data['six-' + str(n)]['spec_y_' + m]
                norm_time = data['six-' + str(n)]['count_time']

                if k == 0:
                    x_ref = x
                    y_ref = y

                x_new = x_ref
                y_new = np.interp(x_ref, x, y) / norm_time

                if shift_type == 'self':
                    sig_plot(x_new + xshift_dict[m][k], y_new, ax=eval('ax' + str(int(i + 1))), i=k,
                             color=eval('colors' + str(int(i + 1)))[k, :])
                elif shift_type == 'L':
                    sig_plot(x_new + xshift_dict['l'][k], y_new, ax=eval('ax' + str(int(i + 1))), i=k,
                             color=eval('colors' + str(int(i + 1)))[k, :])
                elif shift_type == 'R':
                    sig_plot(x_new + xshift_dict['r'][k], y_new, ax=eval('ax' + str(int(i + 1))), i=k,
                             color=eval('colors' + str(int(i + 1)))[k, :])
                else:
                    sig_plot(x_new + xshift_dict['av'][k], y_new, ax=eval('ax' + str(int(i + 1))), i=k,
                             color=eval('colors' + str(int(i + 1)))[k, :])
        ##############################################################
        ax1.axvline(x=E0_pxl_l, linestyle='-', color='k', lw=0.8)
        ax2.axvline(x=E0_pxl_r, linestyle='-', color='k', lw=0.8)

        # ax1.set_title('Left sensor', fontdict={'size': 18})
        ax1.set_xlabel('pixel', fontdict={'size': 10})
        ax1.set_ylabel('photon No./s', fontdict={'size': 10})
        ax1.set_xlim(xlow_l, xhigh_l)

        # ax2.set_title('Right sensor', fontdict={'size': 18})
        ax2.set_xlabel('pixel', fontdict={'size': 10})
        ax2.set_xlim(xlow_r, xhigh_r)

        ax3.set_xlabel('cor_interp No.', fontdict={'size': 10})
        ax3.set_ylabel('shift', fontdict={'size': 10})

        ax4.set_xlabel('scan No.', fontdict={'size': 10})
        ax4.legend()

        # print(
        #     'Correlation for Left Sensor ranges from x={} to x={} and the interpolation for correlation is {}!'.format(
        #         xlow_l, xhigh_l, cor_interp))
        # print('****************************************************************************')
        # print(
        #     'Correlation for Right Sensor ranges from x={} to x={} and the interpolation for correlation is {}!'.format(
        #         xlow_r, xhigh_r, cor_interp))
        #####################################
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################
        
        
    cor_interp_s = IntSlider(min=0, max=35, step=1, value=0, description='cor_interp')
    cor_interp_s.style.handle_color = 'lightgreen'

    xlow_l_s = FloatSlider(min=-5, max=x_trail_l.max(), step=0.1, value=xlow_l_, description='xlow_l')
    xhigh_l_s = FloatSlider(min=0, max=x_trail_l.max() + 10, step=0.1, value=xhigh_l_, description='xhigh_l')
    xlow_r_s = FloatSlider(min=-5, max=x_trail_r.max(), step=0.1, value=xlow_r_, description='xlow_r')
    xhigh_r_s = FloatSlider(min=0, max=x_trail_r.max() + 10, step=0.1, value=xhigh_r_, description='xhigh_r')
    
    shift_type_s = Dropdown(options=[('self', 'self'), ('L', 'L'), ('R', 'R'), ('AV', 'AV')], value='self',
                            description='shift_tpye')

    xlow_l_s.style.handle_color = 'royalblue'
    xhigh_l_s.style.handle_color = 'royalblue'

    xlow_r_s.style.handle_color = 'deeppink'
    xhigh_r_s.style.handle_color = 'deeppink'

    mwidget = ipy_interact(view_sig, {'xlow_l': xlow_l_s, 'xhigh_l': xhigh_l_s,
                                      'xlow_r': xlow_r_s, 'xhigh_r': xhigh_r_s,
                                      'cor_interp': cor_interp_s, 'shift_type': shift_type_s})

    left_box = VBox([xlow_l_s, xhigh_l_s])
    center_box = VBox([xlow_r_s, xhigh_r_s])
    right_box = VBox([cor_interp_s, shift_type_s])

    display(HBox([left_box, center_box, right_box]), mwidget)  # Show all controls

    return {'xlow_l': xlow_l_s, 'xhigh_l': xhigh_l_s,
            'xlow_r': xlow_r_s, 'xhigh_r': xhigh_r_s,
            'cor_interp': cor_interp_s, 'shift_type': shift_type_s}










""
def spec_cor(data, scan=None, sample=None,
             slope_l=0, slope_r=0, points_per_pixel=1,
             cor_roi_l=None, cor_roi_r=None, cor_interp=0, shift_type='self',
             fig=None, plt_close=1):
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))
    
    # specify the border line between left and right sensors!
    # border = data['six-' + str(scan[0])]['image_size'][0]/2
    
    # Specify the energy calibration
    E_cali=data['six-' + str(scan[0])]['E_cali']
    
    xshift_dict, E0_pxl = shift_dict(data, scan=scan,
                                     slope_l=slope_l, slope_r=slope_r, points_per_pixel=points_per_pixel,
                                     cor_roi_l=cor_roi_l, cor_roi_r=cor_roi_r, cor_interp=cor_interp)
    raw_data = data
    ##############################################################
    color_seed = ['royalblue', 'deeppink']
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Data Process', figsize=(10, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        fig = plt.gcf()
        fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        # plt.cla()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    def sig_plot(x, y, ax, color, label, xlim=None):
        # ax.clear()
        ax.plot(x, y, linestyle='-', marker='.', color=color, label=label)
        if xlim is None:
            pass
        else:
            ax.set_xlim(xlim[0], xlim[1])
        # ax.set_xlabel('pixel',fontdict={'size':15})
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

    def xshift_plot(x, y, ax, color):
        ax.plot(x, y, linestyle='-', marker='.', color=color)
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

    sensor = ['l', 'r']

    ############################################################################################################################
    def view_sig(E_cor_low=-200, E_cor_high=1000, cor_interp_E=0,
                 E_low=-800, E_high=2000,
                 points_per_pixel_spec=points_per_pixel,points_per_pixel_shift=points_per_pixel,
                 E_shift=0, y_shift=0, save='No'):
        ##############################################################
        ax1.clear()
        ax2.clear()
        nonlocal sample
        ############################################################################################################################
        data_total = {}
        data_total['cor_roi_E'] = np.array([E_cor_low, E_cor_high])
        data_total['cor_roi'] = np.array([E_cor_low / E_cali + E0_pxl, E_cor_high / E_cali + E0_pxl])

        ##############################################################
        shift_data = sig2spec(raw_data, scan=scan,slope_l=slope_l, slope_r=slope_r,
                        points_per_pixel=points_per_pixel_shift)

    
    
        xshift_array = sensor_shift(data = shift_data, scan=scan, sensor=sensor, xshift_dict=xshift_dict,
                                    shift_type=shift_type,shift_roi=data_total['cor_roi'],cor_interp=cor_interp_E)
        ##############################################################
        data = sig2spec(raw_data, scan=scan, slope_l=slope_l, slope_r=slope_r,
                        points_per_pixel=points_per_pixel_spec)
        ############################################################################################################################
        for i, m in enumerate(sensor):
            for k, n in enumerate(scan):
                norm_time = data['six-' + str(n)]['count_time']
                x = data['six-' + str(n)]['spec_x_' + m]
                y = data['six-' + str(n)]['spec_y_' + m] / norm_time

                if k == 0:
                    x_ref = x
                    y_ref = y
                    data_total['spec_x_' + m] = x_ref
                    data_total['spec_y_' + m] = np.zeros(len(y_ref))
                    ##############################################################

                x_new = x_ref
                y_new = np.interp(x_ref, x, y)
                data['six-' + str(n)]['spec_cor_x_' + m] = x_ref

                if i == 0:
                    data['six-' + str(n)]['cor_shift_ref'] = int(scan[0])
                    data['six-' + str(n)]['shift_type'] = shift_type
                    data['six-' + str(n)]['E_cali'] = E_cali
                ##############################################################
                if shift_type == 'self':
                    data['six-' + str(n)]['spec_cor_y_' + m] = np.interp(x_ref, x_new + xshift_dict[m][k], y_new)
                    data['six-' + str(n)]['cor_shift_' + m] = xshift_dict[m][k]
                elif shift_type == 'L':
                    data['six-' + str(n)]['spec_cor_y_' + m] = np.interp(x_ref, x_new + xshift_dict['l'][k], y_new)
                    data['six-' + str(n)]['cor_shift_' + m] = xshift_dict['l'][k]
                elif shift_type == 'R':
                    data['six-' + str(n)]['spec_cor_y_' + m] = np.interp(x_ref, x_new + xshift_dict['r'][k], y_new)
                    data['six-' + str(n)]['cor_shift_' + m] = xshift_dict['r'][k]
                else:
                    data['six-' + str(n)]['spec_cor_y_' + m] = np.interp(x_ref, x_new + xshift_dict['av'][k], y_new)
                    data['six-' + str(n)]['cor_shift_' + m] = xshift_dict['av'][k]

                # data['six-' + str(n)]['E_' + m] = (x_ref - E0_pxl) * E_cali
                # data['six-' + str(n)]['rixs_' + m] = data['six-' + str(n)]['spec_cor_y_' + m]

                ##############################################################
                data_total['spec_y_' + m] += data['six-' + str(n)]['spec_cor_y_' + m]
            data_total['spec_y_' + m] /= len(scan)  # Normalize the data by time!!!
        for p, q in enumerate(sensor):
            x = data_total['spec_x_' + q]
            y = data_total['spec_y_' + q]

            if p == 0:
                x_ref = x
                y_ref = y
                data_total['spec_x_comb'] = x_ref
                data_total['spec_y_comb'] = np.zeros(len(y_ref))

            x_new = x_ref
            y_new = np.interp(x_ref, x, y)
            
            ############################################################################################################
            data_total['spec_cor_x_' + q] = x_ref
            if p==0:
                data_total['spec_cor_y_' + q] = np.interp(x_ref, x_new, y_new)
            else:
                data_total['spec_cor_y_' + q] = np.interp(x_ref, x_new + xshift_array[-1], y_new)
            ######################################################
            data_total['E_' + q] = (x_ref - E0_pxl) * E_cali + E_shift
            data_total['rixs_' + q] = data_total['spec_cor_y_' + q]
            ######################################################
            data_total['spec_y_comb'] += data_total['spec_cor_y_' + q]
            ##############################################################
            for t in scan:
                local_x = data['six-' + str(t)]['spec_cor_x_' + q]
                local_y = data['six-' + str(t)]['spec_cor_y_' + q]
                data['six-' + str(t)]['spec_comb_x_' + q] = local_x
                                
                if p == 0:
                    local_x_ref = local_x
                    
                if p==0:
                    data['six-' + str(t)]['spec_comb_y_' + q] = np.interp(local_x_ref, local_x, local_y)
                else:
                    data['six-' + str(t)]['spec_comb_y_' + q] = np.interp(local_x_ref, local_x + xshift_array[-1], local_y)
                    
                data['six-' + str(t)]['comb_shift'] = xshift_array[-1]
                
                data['six-' + str(t)]['E_' + q] = (local_x_ref - E0_pxl) * E_cali + E_shift
                data['six-' + str(t)]['rixs_' + q] = data['six-' + str(t)]['spec_comb_y_' + q]
                if p == 0:
                    local_E_ref = data['six-' + str(t)]['E_' + q]
                    data['six-' + str(t)]['rixs'] = np.zeros(len(local_E_ref))
                data['six-' + str(t)]['E'] = local_E_ref
                data['six-' + str(t)]['rixs'] += np.interp(local_E_ref, data['six-' + str(t)]['E_' + q],
                                                           data['six-' + str(t)]['rixs_' + q])
                if p == 1:
                    data['six-' + str(t)]['rixs'] /= 2
                data['six-' + str(t)]['scan'] = t
            ##############################################################
            sig_plot(data_total['E_' + q], data_total['rixs_' + q] + y_shift * p, ax=ax1, color=color_seed[p],
                     label=q + ' sensor', xlim=[E_low, E_high])
            if p != 0:
                xshift_plot(np.arange(0, cor_interp_E + 1, 1), xshift_array * E_cali, ax=ax2, color='k')

        data_total['E'] = (data_total['spec_x_comb'] - E0_pxl) * E_cali + E_shift
        data_total['rixs'] = data_total['spec_y_comb']

        data_total['E0_pxl'] = E0_pxl
        data_total['E_cali'] = E_cali

        data_total['cor_shift_E'] = xshift_array[-1] * E_cali
        data_total['shift_ref'] = 'left sensor!'
        data_total['cor_interp'] = cor_interp_E
        data_total['scan'] = scan
        data_total['shift_type'] = shift_type
        data_total['points_per_pixel_shift'] = points_per_pixel_shift

        data_total.update(meta_data(data, scan=scan))

        ##############################################################
        ax1.plot(data_total['E'], data_total['rixs'] + y_shift * 2, linestyle='-', color='seagreen', marker='o',
                 mfc='w',mec='seagreen', label='combine')
        ax1.axvline(0, linestyle='-', color='k', lw=0.8)
        ax1.set_xlabel('energy/meV', fontdict={'size': 15})
        ax1.set_ylabel('photon No./s', fontdict={'size': 15})

        ax1.set_xlim(E_low, E_high)
        ax1.legend()

        ax2.set_xlabel('cor_interp_E No.', fontdict={'size': 15})
        ax2.set_ylabel('energy shift/meV', fontdict={'size': 15})

        # print('Correlation ranges from Ei={}meV to Ef={}meV and the interpolation for correlation is {}!'.format(E_cor_low, E_cor_high, cor_interp_E))

        if save == 'No':
            pass
        else:
            if data_total['points_per_pixel'].mean() < 0.09:
                points_per_pixel_str = '{:.2f}'.format(data_total['points_per_pixel'].mean())
            elif (data_total['points_per_pixel'].mean() > 0.09) and (data_total['points_per_pixel'].mean() < 0.98):
                points_per_pixel_str = '{:.1f}'.format(data_total['points_per_pixel'].mean())
            else:
                if data_total['points_per_pixel'].mean() in np.arange(1.5,7.6,1):
                    points_per_pixel_str = '{:.1f}'.format(data_total['points_per_pixel'].mean())
                else:
                    points_per_pixel_str = '{:.0f}'.format(data_total['points_per_pixel'].mean())
                
                
            save_folder = cwd + '/Data/' + 'points_per_pixel' + '-' + points_per_pixel_str + '/'
            save_data(data, save_folder + 'RawData/', scan=scan, data_format=save, sample=sample)
            save_data_total(data_total, save_folder, data_format=save, sample=sample)
            if sample==None:
                sample='six'
            if len(scan)==1:
                print(sample+'-'+str(int(scan[0]))+' has been saved into {} files --> points_per_pixel={}!!!'.format(save.upper(),points_per_pixel_str))
            else:
                print(sample+'-'+str(int(scan[0]))+'_'+str(int(scan[-1]))+' has been saved into {} files --> points_per_pixel={}!!!'.format(save.upper(),points_per_pixel_str))

                
        #####################################
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################
    ####################################################################################################################
    cor_interp_E_s = IntSlider(min=0, max=30, step=1, value=0, description='cor_interp_E')
    cor_interp_E_s.style.handle_color = 'lightgreen'

    E_cor_low_s = FloatSlider(min=-3000, max=8000, step=1, value=-200, description='E_cor_low')
    E_cor_high_s = FloatSlider(min=-3000, max=8000, step=1, value=400, description='E_cor_high')

    E_low_s = FloatSlider(min=-3000, max=12000, step=1, value=-200, description='E_low')
    E_high_s = FloatSlider(min=-3000, max=12000, step=1, value=400, description='E_high')

    ####################################################################################################################

    points_per_pixel_spec_value = np.hstack(
        (np.arange(0.02, 0.084, 0.02), np.arange(0.1, 0.95, 0.1), np.arange(1, 8.1, 0.5)))
    points_per_pixel_spec_s = SelectionSlider(options=[("%g" % t, t) for t in points_per_pixel_spec_value],
                                              value=points_per_pixel,
                                              description='points_per_pixel_spec',
                                              style={'description_width': 'initial'})
    points_per_pixel_spec_s.style.handle_color = 'blue'

    points_per_pixel_shift_value = np.hstack(
        (np.arange(0.02, 0.084, 0.02), np.arange(0.1, 0.95, 0.1), np.arange(1, 9, 1)))
    points_per_pixel_shift_s = SelectionSlider(options=[("%g" % t, t) for t in points_per_pixel_shift_value],
                                              value=points_per_pixel,
                                              description='points_per_pixel_shift',
                                              style={'description_width': 'initial'})
    points_per_pixel_shift_s.style.handle_color = 'orange'

    ####################################################################################################################

    E_shift_s = FloatSlider(min=-500, max=500, step=0.1, value=0, description='E_shift')
    y_shift_s = FloatSlider(min=0, max=5, step=0.05, value=0, description='y_shift')
    save_s = Dropdown(options=[('Yes_hdf', 'hdf'), ('Yes_txt', 'txt'), ('No', 'No')], value='No', description='save!!!')

    
    mwidget = ipy_interact(view_sig,
                           {'E_cor_low': E_cor_low_s, 'E_cor_high': E_cor_high_s, 'cor_interp_E': cor_interp_E_s,
                            'E_low': E_low_s, 'E_high': E_high_s,
                            'points_per_pixel_spec': points_per_pixel_spec_s,'points_per_pixel_shift':points_per_pixel_shift_s,
                            'E_shift': E_shift_s, 'y_shift': y_shift_s, 'save': save_s})

    left_but = Button(description='Correlation Shift between two sensors',layout=Layout(width='auto', grid_area='header'),
                      style=ButtonStyle(button_color='honeydew'))
    left_box = VBox([left_but,E_cor_low_s, E_cor_high_s, cor_interp_E_s,points_per_pixel_shift_s],layout=Layout(
        width='auto',
        flex_flow='column',
        display='flex',
        border='solid 2px',
        align_items='stretch'))
    
    cen_but = Button(description='Energy Offset',layout=Layout(width='auto', grid_area='header'),
                      style=ButtonStyle(button_color='lavender'))
    center_box = VBox([cen_but,E_shift_s,E_low_s, E_high_s,y_shift_s],layout=Layout(
        width='auto',
        flex_flow='column',
        border='solid 2px',
        align_items='stretch'))

    save_all_button = Button(description='Save All points_per_pixel (Click)', layout=Layout(width='auto', grid_area='header'),
                                 style=ButtonStyle(button_color='bisque'))
    save_all_button.on_click(lambda x: save_all())
    
    right_but = Button(description='Save Data',layout=Layout(width='auto', grid_area='header'),
                      style=ButtonStyle(button_color='bisque'))
    right_box = VBox([right_but,points_per_pixel_spec_s, save_all_button, save_s],layout=Layout(
        width='auto',
        flex_flow='column',
        border='solid 2px',
        align_items='stretch'))
    
    display(HBox([left_box, center_box, right_box]), mwidget)  # Show all controls

    def save_all():
        for points_per_pixel_spec in points_per_pixel_spec_value:
            view_sig(E_cor_low=E_cor_low_s.value, E_cor_high=E_cor_high_s.value, cor_interp_E=cor_interp_E_s.value, E_low=E_low_s.value, 
                     E_high=E_high_s.value,points_per_pixel_spec=points_per_pixel_spec,points_per_pixel_shift=points_per_pixel_shift_s.value,E_shift=E_shift_s.value, y_shift=y_shift_s.value, save='hdf')


""
def sensor_shift(data, scan, sensor, xshift_dict, shift_type, shift_roi,cor_interp):
    data_total = {}
    #####################################################################################
    for i, m in enumerate(sensor):
        for k, n in enumerate(scan):
            norm_time = data['six-' + str(n)]['count_time']
            x = data['six-' + str(n)]['spec_x_' + m]
            y = data['six-' + str(n)]['spec_y_' + m] / norm_time

            if k == 0:
                x_ref = x
                y_ref = y
                data_total['spec_x_' + m] = x_ref
                data_total['spec_y_' + m] = np.zeros(len(y_ref))
                ##############################################################

            x_new = x_ref
            y_new = np.interp(x_ref, x, y)
            data['six-' + str(n)]['spec_cor_x_' + m] = x_ref

            ##############################################################
            if shift_type == 'self':
                data['six-' + str(n)]['spec_cor_y_' + m] = np.interp(x_ref, x_new + xshift_dict[m][k], y_new)
                data['six-' + str(n)]['cor_shift_' + m] = xshift_dict[m][k]
            elif shift_type == 'L':
                data['six-' + str(n)]['spec_cor_y_' + m] = np.interp(x_ref, x_new + xshift_dict['l'][k], y_new)
                data['six-' + str(n)]['cor_shift_' + m] = xshift_dict['l'][k]
            elif shift_type == 'R':
                data['six-' + str(n)]['spec_cor_y_' + m] = np.interp(x_ref, x_new + xshift_dict['r'][k], y_new)
                data['six-' + str(n)]['cor_shift_' + m] = xshift_dict['r'][k]
            else:
                data['six-' + str(n)]['spec_cor_y_' + m] = np.interp(x_ref, x_new + xshift_dict['av'][k], y_new)
                data['six-' + str(n)]['cor_shift_' + m] = xshift_dict['av'][k]

            ##############################################################
            data_total['spec_y_' + m] += data['six-' + str(n)]['spec_cor_y_' + m]
        data_total['spec_y_' + m] /= len(scan)  # Normalize the data by time!!!
    ##############################################################
    for p, q in enumerate(sensor):
        x = data_total['spec_x_' + q]
        y = data_total['spec_y_' + q]

        if p == 0:
            x_ref = x
            y_ref = y
            data_total['spec_x_comb'] = x_ref
            data_total['spec_y_comb'] = np.zeros(len(y_ref))

        x_new = x_ref
        y_new = np.interp(x_ref, x, y)
        xshift_array = np.array([])
        for cor_num in np.arange(0, cor_interp + 1, 1):
            if cor_num == 0:
                xshift = 0
            else:
                xshift = sig_cor(x_ref=x_ref, y_ref=y_ref, x=x_new, y=y_new, roi_cor=shift_roi,
                                 cor_interp=int(cor_num))
            xshift_array = np.append(xshift_array, xshift)

    return xshift_array
""
def shift_dict(data, scan=None, slope_l=0, slope_r=0, points_per_pixel=1,
               cor_roi_l=0, cor_roi_r=400, cor_interp=0):
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))

    data = sig2spec(data, scan=scan, slope_l=slope_l, slope_r=slope_r, points_per_pixel=points_per_pixel)

    ##############################################################
    sensor = ['l', 'r']
    ##############################################################
    xshift_dict = {}
    for i, m in enumerate(sensor):
        xshift_dict[m] = np.array([])
        cor_roi = eval('cor_roi_' + m)
        for k, n in enumerate(scan):
            x = data['six-' + str(n)]['spec_x_' + m]
            y = data['six-' + str(n)]['spec_y_' + m]
            if k == 0:
                x_ref = x
                y_ref = y
                ##############################################################
                # Set-up the E0_pxl_l and E0_pxl_r!!!
                if m == 'l':
                    E0_pxl = x_ref[y_ref[(x_ref >= cor_roi[0]) & (x_ref <= cor_roi[1])].argmax()] + cor_roi[0]
                ##############################################################
            x_new = x_ref
            y_new = np.interp(x_ref, x, y)
            if cor_interp == 0:
                xshift = 0
            else:
                xshift = sig_cor(x_ref=x_ref, y_ref=y_ref, x=x_new, y=y_new, roi_cor=cor_roi, cor_interp=cor_interp)

            xshift_dict[m] = np.append(xshift_dict[m], xshift)
    xshift_dict['av'] = (xshift_dict['l'] + xshift_dict['r']) / 2

    return xshift_dict, E0_pxl





""
def rixs1d(data_folder, scan, sample=None, data_type='hdf', fig=None, plt_close=1):
    if sample is None:
        sample = 'six'
    
    ##############################################################
    # Extract the points_per_pixel from data_folder
    points_per_pixel_string = 'False'
    for t in data_folder.split('/'):
        if (len(t)==18) or (len(t)==20) or (len(t)==21):
            if t[:16]=='points_per_pixel':
                points_per_pixel_string = t[17:]
    ##############################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Display RIXS Spectra', figsize=(6, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(6.5, 5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax = fig.add_subplot(111)
    else:
        # fig = plt.gcf()
        # fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        # plt.cla()
        ax = fig.add_subplot(111)
    colors = cm.winter(np.linspace(0, 1, len(scan)))
    # colors = cycle(('royalblue','deeppink','seagreen','olive','darkorange',
                    # 'brown','dodgerblue','darkviolet','teal','slategrey','tomato','slateblue'))
    ##############################################################
    def view_sig(sig='rixs', legend='off', errorbar='off',xshift=0, yshift=0,
                 x_low=-500, x_high=2300, y_low=0, y_high=0):
        
        nonlocal points_per_pixel_string
        
        ax.clear()

        if sig == 'rixs':
            sig_x = 'E'
            sig_path = 'comb'
        elif sig == 'rixs_l':
            sig_x = 'E_l'
            sig_path = 'left'
        elif sig == 'rixs_r':
            sig_x = 'E_r'
            sig_path = 'right'
        ####################################################################
        for i, n in enumerate(scan):
            if data_type == 'hdf':
                list_of_files = globf(data_folder + sample+'-' + str(n) + '*.hdf')
                latest_file = max(list_of_files, key=path.getctime)
                data_file = h5_file(latest_file, 'r')
                print(latest_file)

                count_time = np.sum(data_file['meta']['count_time'][:,0])
                ####################################################################
                if len(list(data_file.keys())) == 3:
                    if sig_path == 'comb':
                        x = data_file['data']['E'][:, 0]
                        y = data_file['data']['rixs'][:, 0]
                    else:
                        x = data_file['data'][sig_path]['E'][:, 0]
                        y = data_file['data'][sig_path]['rixs'][:, 0]

                elif len(list(data_file.keys())) == 2:
                    x = data_file['data'][sig_x][:, 0]
                    y = data_file['data'][sig][:, 0]
                else:
                    print('Data structure is wrong, please process the data again!!!')
                ##############################################################################
                if i==0:
                    x_ref = x
                    y_ref = y
                y_interp = np.interp(x_ref,x,y)
                y_interp_error = np.sqrt(y_interp / float(points_per_pixel_string) * count_time)*float(points_per_pixel_string)/count_time
    
                ####################################################################
                if errorbar=='on':
                    ax.errorbar(x_ref + xshift * i,y_interp+yshift*i,yerr=y_interp_error,
                                 marker='.',mfc=colors[i,:],mec=colors[i,:],ecolor=colors[i,:],markersize=5,markeredgewidth=1,
                                 linestyle='-',color=colors[i,:],lw=1.0,elinewidth=1,capsize=2.,
                                 label='six-{:.0f}'.format(n))
                else:
                    ax.plot(x_ref + xshift * i, y_interp + yshift * i, linestyle='-', color=colors[i,:], marker='.', mfc=colors[i,:],
                            label='six-' + str(n))

            else:
                list_of_files = globf(data_folder + sample + '-' + str(n) + '*.' + data_type)
                latest_file = max(list_of_files, key=path.getctime)
                data_file = np.genfromtxt(latest_file, skip_header=1)
                print(latest_file)
                ##################################
                if sig_path == 'comb':
                    x = data_file[:, 0]
                    y = data_file[:, 1]
                elif sig_path == 'left':
                    x = data_file[:, 2]
                    y = data_file[:, 3]
                else:
                    x = data_file[:, 4]
                    y = data_file[:, 5]
                ####################################################################
                ax.plot(x + xshift * i, y + yshift * i, linestyle='-', color=colors[i,:], marker='.', mfc=colors[i,:],
                        label='six-' + str(n))

        ax.axvline(x=0, linestyle='-', color='k', lw=0.6)
        ax.axhline(y=0, linestyle='--', color='k', lw=0.6)

        if x_high <= x_low:
            x_high = x_low + 500
        ax.set_xlim(x_low, x_high)
            
        if (y_high==0):
            ax.set_ylim(y_low,)
        else:
            if y_high <= y_low:
                y_high = y_low + 3
            ax.set_ylim(y_low, y_high)
        ####################################################################
        ax.set_xlabel('energy loss/meV', fontdict={'size': 13})
        ax.set_ylabel('photons/s ('+sig+')', fontdict={'size': 13})
        
        ax.grid(linestyle='-.')
        ####################################################################
        if legend == 'on':
            ax.legend()
        else:
            pass
        ####################################################################
        if points_per_pixel_string == 'False':
            pass
        else:
            ax.text(.85, .68, str(points_per_pixel_string)+'p/pixel', transform=ax.transAxes,# ha="left", va="center",
                   bbox=dict(edgecolor='green', facecolor='w',alpha=0.8))
            
        #####################################
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################
        ####################################################################
    # data_type_s = Dropdown(options=[('hdf', 'hdf'), ('txt', 'txt')], value=data_type, description='data_type')
    sig_s = Dropdown(options=[('rixs', 'rixs'), ('rixs_l', 'rixs_l'), ('rixs_r', 'rixs_r')], value='rixs',
                     description='sig')
    legend_s = Dropdown(options=[('off', 'off'), ('on', 'on')], value='off', description='legend')
    errorbar_s = Dropdown(options=[('off', 'off'), ('on', 'on')], value='off', description='errorbar')

    xshift_s = FloatSlider(min=-1000, max=1000, step=0.1, value=0, description='xshift')
    yshift_s = FloatSlider(min=0, max=50, step=0.01, value=0, description='yshift')

    x_low_s = FloatSlider(min=-5000, max=25000, step=1, value=-500, description='x_low')
    x_high_s = FloatSlider(min=-5000, max=25000, step=1, value=2300, description='x_high')

    y_low_s = FloatSlider(min=-10, max=20, step=0.1, value=0, description='y_low')
    y_high_s = FloatSlider(min=-10, max=20, step=0.1, value=0, description='y_high')

    mwidget = ipy_interact(view_sig, {'sig': sig_s, 'legend': legend_s, 'errorbar': errorbar_s,
                                      'xshift': xshift_s, 'yshift': yshift_s,
                                      'x_low': x_low_s, 'x_high': x_high_s,
                                      'y_low': y_low_s, 'y_high': y_high_s, })

    left_box = VBox([sig_s, legend_s,errorbar_s])
    center_box1 = VBox([xshift_s, x_low_s, x_high_s])
    right_box = VBox([yshift_s, y_low_s, y_high_s])

    display(HBox([center_box1, right_box, left_box]), mwidget)  # Show all controls


""
def rixs2d(data_folder, scan, sample=None, sig='rixs', vari='th', vari_offset=0,
           cut_type='V', colormap='jet', fig=None, plt_close=1):
    ##############################################################
    if sample is None:
        sample = 'six'

    vari_array = np.array([])
    I_max = 0
    for i, n in enumerate(scan):
        list_of_files = globf(data_folder + sample+'-' + str(n) + '*.hdf')
        latest_file = max(list_of_files, key=path.getctime)
        data_file = h5_file(latest_file, 'r')
        print(latest_file)
        ##########################################

        if i == 0:
            E2d_st = data_file['data']['E'][:, 0]
        local_I = (data_file['data']['rixs'][:, 0]).max()
        if I_max < local_I:
            I_max = local_I
        vari_array = np.append(vari_array, data_file['meta'][vari][0, 0] + vari_offset)
    scan_sort = scan[vari_array.argsort()]
    vari_array_sort = vari_array[vari_array.argsort()]
    ##############################################################
    vari_step_st = vari_array_sort[1] - vari_array_sort[0]
    e_step_st = E2d_st[1] - E2d_st[0]
    ##############################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Display RIXS Spectra', figsize=(10, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(10, 4.5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax1 = fig.add_subplot(121)
        cbar_ax = fig.add_axes([0.095, 0.73, 0.012, 0.15])  # pos=[left, bottom, width, height]
        ax2 = fig.add_subplot(122)
    else:
        # fig = plt.gcf()
        # fig.set_size_inches(10, 4.5)
        # fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        # fig.patch.set_alpha(0.1)
        # plt.cla()
        ax1 = fig.add_subplot(121)
        cbar_ax = fig.add_axes([0.095, 0.73, 0.012, 0.15])  # pos=[left, bottom, width, height]
        ax2 = fig.add_subplot(122)

    ##############################################################
    def img_plot(spec2d, E2d, vari_array, ax, colormap, zlim):
        vari_array_mesh,E2d_mesh = np.meshgrid(vari_array,E2d)
        im = ax.pcolormesh(vari_array_mesh,E2d_mesh,np.fliplr(np.rot90(spec2d, k=-1)),
                           shading='nearest',cmap=colormap, # edgecolors='k',linewidths=0.5,
                           vmin=zlim[0], vmax=zlim[1])
        
        # im = ax.imshow(np.fliplr(np.rot90(spec2d, k=-1)), aspect='auto', interpolation='none', origin='lower',
        #                vmin=zlim[0], vmax=zlim[1])
        # im.set_extent([vari_array.min(), vari_array.max(), E2d.min(), E2d.max()])
        # im.set_cmap(colormap)
        # ax.plot(np.array([-100, 200]), [0, 0], '--w', linewidth=1)
        ax.axhline(0,linestyle='--',color='w',lw=1)
        return im
    #################################################################
    def sig_plot(x, y, ax):
        ax.plot(x, y, linestyle='-', marker='.', color='k')
        ax.axes.grid(color='k', linestyle='--', linewidth=1, alpha=0.3)

    ##############################################################
    vari_step = vari_array_sort[1] - vari_array_sort[0]

    if cut_type == 'V':
        cut_0 = vari_array_sort[0]
    elif cut_type == 'H':
        cut_0 = 0

    ##############################################################
    def view_sig(h_low=vari_array_sort.min(), h_high=vari_array_sort.max(), sig='rixs',
                 v_low=-500, v_high=5000, c_low=0, c_high=I_max, E_shift=0, save='No',
                 cut=cut_0, cut_int_wid=0, sig_interp=0, x_low=-500, x_high=5000):

        nonlocal cut_type
        ax1.clear()
        ax2.clear()
        cbar_ax.clear()
        ##############################################################
        if sig == 'rixs':
            sig_x = 'E'
            sig_path = 'comb'
        elif sig == 'rixs_l':
            sig_x = 'E_l'
            sig_path = 'left'
        elif sig == 'rixs_r':
            sig_x = 'E_r'
            sig_path = 'right'

        for i, n in enumerate(scan_sort):
            list_of_files = globf(data_folder + sample+'-' + str(n) + '*.hdf')
            latest_file = max(list_of_files, key=path.getctime)
            data_file = h5_file(latest_file, 'r')
            
            ############################################
            if len(list(data_file.keys())) == 3:
                if sig_path == 'comb':
                    x = data_file['data']['E'][:, 0]
                    y = data_file['data']['rixs'][:, 0]
                else:
                    x = data_file['data'][sig_path]['E'][:, 0]
                    y = data_file['data'][sig_path]['rixs'][:, 0]

            elif len(list(data_file.keys())) == 2:
                x = data_file['data'][sig_x][:, 0]
                y = data_file['data'][sig][:, 0]
            else:
                print('Data structure is wrong, please process the data again!!!')

            if i == 0:
                spec2d = np.zeros([len(scan_sort), len(x)])
                E2d = x
            spec2d[i, :] = np.interp(E2d, x, y)

        #####################################################
        e_step = E2d[1] - E2d[0]

        if cut_type == 'V':
            cut_0 = vari_array_sort[0]
            cut_int_wid_0 = 0

        elif cut_type == 'H':
            cut_0 = 0
            cut_int_wid_0 = e_step

        ##############################################################

        if h_high <= h_low:
            h_high = h_low + (vari_step) * 3

        if v_high <= v_low:
            v_high = v_low + 100

        if c_high <= c_low:
            c_low = spec2d.min()
            c_high = spec2d.max()

        if x_high <= x_low:
            if cut_type == 'V':
                x_high = x_low + 100
            else:
                x_high = x_low + (vari_step) * 3

        ##############################################################
        E2d_shift = E2d + E_shift
        im = img_plot(spec2d=spec2d, E2d=E2d_shift, vari_array=vari_array_sort, ax=ax1, colormap=colormap,
                 zlim=[c_low, c_high])
        
        ####################################################################################
        # Set the colorbar
        # fig=plt.gcf()
        cb = fig.colorbar(im, cax=cbar_ax) # , ticks=[c_low,c_high]
        cb.set_ticks([c_low,c_high])
        cb.ax.set_yticklabels(['{:.2f}'.format(t) for t in [c_low,c_high]],fontsize=10, color='w')
        cb.outline.set_edgecolor('w')
        ####################################################################################

        if cut_type == 'V':
            cut_pos = np.where(np.isclose(vari_array_sort, cut))[0]
            ax1.axvline(x=cut, linestyle='--', color='y', lw=1)
            if cut_int_wid == 0:
                sig_ = spec2d[cut_pos, :][0, :]
                int_roi = np.array([cut, cut])
            else:
                if (cut_pos + int(cut_int_wid)) >= len(vari_array_sort):
                    cut_int_wid = len(vari_array_sort) - cut_pos - 1
                else:
                    pass
                ax1.axvline(x=vari_array_sort[cut_pos + int(cut_int_wid)], linestyle='--', color='y', lw=1)

                int_roi = np.where(
                    (vari_array_sort >= cut) & (vari_array_sort <= (vari_array_sort[cut_pos + int(cut_int_wid)])))[0]
                sig_ = np.sum(spec2d[int_roi, :], axis=0)
            if sig_interp == 0:
                E2d_itp = E2d_shift
                sig_itp = sig_
            else:
                E2d_itp, sig_itp = m_interpolate(E2d_shift, sig_, len(sig_) * int(sig_interp), order=3)
            sig_plot(E2d_itp, sig_itp, ax=ax2)
            ax2.axvline(x=0, linestyle='-', color='r', lw=0.6)
            ax2.axhline(y=0, linestyle='--', color='r', lw=0.4)
            ax2.set_xlabel('Energy (meV)', fontdict={'size': 10})
            ax2.set_title('EDC', fontdict={'size': 18})
            ##############################################################
            if save == 'hdf':
                save_dict = {'spec': spec2d, 'spec_x': vari_array_sort, 'spec_y': E2d_shift, 'spec_y_shift': E_shift,
                             'cut_x': E2d_itp, 'cut_y': sig_itp, 'cut_interp_num': sig_interp,
                             'cut_type': cut_type + ' cut!!!',
                             'cut_wid': '{}=[{:.2f}, {:.2f}]'.format(vari, vari_array_sort[int_roi.min()],
                                                                     vari_array_sort[int_roi.max()]),
                             'scan': scan_sort, 'vari': vari, 'data_channel': sig}

                save_map(save_dict, save_folder=data_folder, data_format='hdf')
                print('Data has been saved into {} files!!!'.format(save.upper()))
            elif save == 'txt':
                save_dict = {'spec': spec2d, 'spec_x': vari_array_sort, 'spec_y': E2d_shift, 'spec_y_shift': E_shift,
                             'cut_x': E2d_itp, 'cut_y': sig_itp, 'cut_interp_num': sig_interp,
                             'cut_type': cut_type + ' cut!!!',
                             'cut_wid': '{}=[{:.2f}, {:.2f}]'.format(vari, vari_array_sort[int_roi.min()],
                                                                     vari_array_sort[int_roi.max()]),
                             'scan': scan_sort, 'vari': vari, 'data_channel': sig}

                save_map(save_dict, save_folder=data_folder, data_format='txt')
                print('Data has been saved into {} files!!!'.format(save.upper()))
            else:
                pass
            ##############################################################

        elif cut_type == 'H':
            ax1.axhline(y=cut, linestyle='--', color='y', lw=1)
            ax1.axhline(y=cut + cut_int_wid, linestyle='--', color='y', lw=1)
            if cut_int_wid == 0:
                sig_ = spec2d[:, (np.abs(E2d_shift - cut)).argmin()]
                int_roi = np.array([(np.abs(E2d_shift - cut)).argmin(), (np.abs(E2d_shift - cut)).argmin()])

            else:
                int_roi = np.where((E2d_shift >= cut) & (E2d_shift <= (cut + cut_int_wid)))[0]
                sig_ = np.sum(spec2d[:, int_roi], axis=1)

            if sig_interp == 0:
                vari_array_itp = vari_array_sort
                sig_itp = sig_
            else:
                vari_array_itp, sig_itp = m_interpolate(vari_array_sort, sig_, len(sig_) * sig_interp, order=3)
            sig_plot(vari_array_itp, sig_itp, ax=ax2)
            ax2.set_xlabel(vari, fontdict={'size': 15})

            ##############################################################
            if save == 'hdf':
                save_dict = {'spec': spec2d, 'spec_x': vari_array_sort, 'spec_y': E2d_shift, 'spec_y_shift': E_shift,
                             'cut_x': vari_array_itp, 'cut_y': sig_itp, 'cut_interp_num': sig_interp,
                             'cut_type': cut_type + ' cut!!!',
                             'cut_wid': 'E=[{:.2f}, {:.2f}]'.format(E2d_shift[int_roi.min()],
                                                                    E2d_shift[int_roi.max()]),
                             'scan': scan_sort, 'vari': vari, 'data_channel': sig}

                save_map(save_dict, save_folder=data_folder, data_format='hdf')
                print('Data has been saved into {} files!!!'.format(save.upper()))
            elif save == 'txt':
                save_dict = {'spec': spec2d, 'spec_x': vari_array_sort, 'spec_y': E2d_shift, 'spec_y_shift': E_shift,
                             'cut_x': vari_array_itp, 'cut_y': sig_itp, 'cut_interp_num': sig_interp,
                             'cut_type': cut_type + ' cut!!!',
                             'cut_wid': 'E=[{:.2f}, {:.2f}]'.format(E2d_shift[int_roi.min()],
                                                                    E2d_shift[int_roi.max()]),
                             'scan': scan_sort, 'vari': vari, 'data_channel': sig}

                save_map(save_dict, save_folder=data_folder, data_format='txt')
                print('Data has been saved into {} files!!!'.format(save.upper()))
            else:
                pass
            ##############################################################

        ax1.set_xlim(h_low, h_high)
        ax1.set_ylim(v_low, v_high)

        ax1.set_xlabel(vari, fontdict={'size': 15})
        ax1.set_ylabel('Energy (meV)', fontdict={'size': 15})

        ax2.set_ylabel('Inten (photon/s)', fontdict={'size': 15})
        ax2.set_xlim(x_low, x_high)
        #####################################
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ##############################################################
        
        
    h_low_s = FloatSlider(min=vari_array_sort.min() - vari_step_st, max=vari_array_sort.max() + vari_step_st,
                          step=0.1, value=vari_array_sort.min(), description='h_low', orientation='vertical')
    h_high_s = FloatSlider(min=vari_array_sort.min() - vari_step_st, max=vari_array_sort.max() + vari_step_st,
                           step=0.1, value=vari_array_sort.max(), description='h_high', orientation='vertical')

    v_low_s = FloatSlider(min=E2d_st.min(), max=E2d_st.max(),
                          step=0.1, value=-500, description='v_low', orientation='vertical')
    v_high_s = FloatSlider(min=E2d_st.min(), max=E2d_st.max(),
                           step=0.1, value=5000, description='v_high', orientation='vertical')

    c_low_s = FloatSlider(min=-I_max * 3, max=I_max * 3,
                          step=0.1, value=0, description='c_low', orientation='vertical')
    c_high_s = FloatSlider(min=-I_max * 3, max=I_max * 3,
                           step=0.1, value=I_max, description='c_high', orientation='vertical')
    E_shift_s = FloatSlider(min=-50, max=50, step=0.1, value=0, description='E_shift')

    sig_s = Dropdown(options=[('rixs', 'rixs'), ('rixs_l', 'rixs_l'), ('rixs_r', 'rixs_r')], value='rixs',
                     description='sig')

    save_s = Dropdown(options=[('Yes_hdf', 'hdf'), ('Yes_txt', 'txt'), ('No', 'No')], value='No', description='save!!!')

    if cut_type == 'V':
        cut_s = SelectionSlider(options=[t for t in vari_array_sort],
                                value=vari_array_sort[0], description='cut', readout=True,
                                readout_format='.2f')
        cut_int_wid_s = IntSlider(min=0, max=len(vari_array_sort) - 1, step=1, value=0, description='cut_int_wid')

        x_low_s = FloatSlider(min=E2d_st.min(), max=E2d_st.max(), step=0.1, value=-500, description='x_low')
        x_high_s = FloatSlider(min=E2d_st.min(), max=E2d_st.max(), step=0.1, value=5000, description='x_high')

    else:
        cut_s = FloatSlider(min=E2d_st.min(), max=E2d_st.max(),
                            step=0.1, value=0, description='cut')
        cut_int_wid_s = FloatSlider(min=0, max=(E2d_st.max() - E2d_st.min()),
                                    step=e_step_st, value=0, description='cut_int_wid')

        x_low_s = FloatSlider(min=vari_array_sort.min() - vari_step_st, max=vari_array_sort.max() + vari_step_st,
                              step=0.1, value=vari_array_sort.min(), description='x_low')
        x_high_s = FloatSlider(min=vari_array_sort.min() - vari_step_st, max=vari_array_sort.max() + vari_step_st,
                               step=0.1, value=vari_array_sort.max(), description='x_high')

    sig_interp_s = IntSlider(min=0, max=6, step=1, value=0, description='sig_interp')

    mwidget = ipy_interact(view_sig, {'h_low': h_low_s, 'h_high': h_high_s, 'c_low': c_low_s, 'sig': sig_s,
                                      'v_low': v_low_s, 'v_high': v_high_s, 'c_high': c_high_s, 'save': save_s,
                                      'cut': cut_s, 'cut_int_wid': cut_int_wid_s, 'E_shift': E_shift_s,
                                      'x_low': x_low_s, 'x_high': x_high_s, 'sig_interp': sig_interp_s})

    left_box = HBox([v_low_s, v_high_s, h_low_s, h_high_s, c_low_s, c_high_s])
    center_box2 = VBox([x_low_s, x_high_s, sig_interp_s, save_s])
    right_box = VBox([sig_s, E_shift_s, cut_s, cut_int_wid_s])

    display(HBox([left_box, right_box, center_box2]), mwidget)  # Show all controls

    

""
def meta_plot(data_folder, scan, sample=None, meta='th',fig=None, plt_close=1):
    scan_sort = np.sort(scan)
    ##############################################################
    if sample is None:
        sample = 'six'
        
        
    ##############################################################
    # Initialize the figure properties
    if fig is None:
        if plt_close == 1:
            plt.close('all')
            fig = plt.figure('Display Meta-data', figsize=(6.5, 4.5))
        else:
            fig = plt.figure(strftime("%Y %b %d  %H:%M:%S"), figsize=(6.5, 5))
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        ax = fig.add_subplot(111)
    else:
        # fig = plt.gcf()
        # fig.set_size_inches(10, 4.5)
        fig.patch.set_facecolor((0.1, 0.8, 0.8, 0.03))
        fig.patch.set_alpha(0.1)
        # plt.cla()
        ax = fig.add_subplot(111)
        
    ######################################################
    vari_data =  np.array([])
    for i, n in enumerate(scan_sort):
        try:
            data_file = h5_file(data_folder + sample+'-' + str(n)+ '.hdf', 'r')
        except:
            data_file = h5_file(globf(data_folder + sample+'-' +str(n) + '_' + '*.hdf')[-1], 'r')
            # print(globf(data_folder +  sample+'-' +str(n) + '_' + '*.hdf')[-1])
        ##############################################################################
        try:
            vari_data = np.append(vari_data,data_file['meta'][meta][0,0])
        except:
            vari_data = np.append(vari_data,data_file['meta'][meta][0])
    ##############################################################################
    if type(vari_data[0]) is np.str_:
        print('***************************************************************************')
        print('The type of this meta data is string, not number!!! So the plot is skipped!')
        print('***************************************************************************')
    else:
        ax.plot(scan_sort,vari_data,
                 marker='o',markersize=5,mfc='w',mec='k',linestyle='-', color='k',lw=1) 
    ##############################################################################
    ax.grid(linestyle='-.')
    ax.set_xlabel('scan No.',fontdict={'size': 15})
    ax.set_ylabel(meta,fontdict={'size': 15})


import os
import numpy as np
from scipy import interpolate
from scipy.signal import correlate
from math import factorial
# import pandas as pd
from pandas import concat as pd_concat
from pandas import DataFrame as pd_df
from prettytable import PrettyTable
from time import (time, ctime)
from h5py import File as h5_file
from h5py import special_dtype
from glob import glob as globf
###############################################
# from databroker import DataBroker
# db = DataBroker.named('six')

from databroker import Broker
db = Broker.named('six')


def six_data(scan, meta=None,E_cali=21.77):
    if meta is None:
        meta = np.array(['cryo_x', 'cryo_y', 'cryo_z', 'cryo_t', 'pgm_en','oc_twoth','epu1_gap_readback',
                         'stemp_temp_B_T','stemp_temp_A_T', 'epu1_phase_readback', 'extslt_vg', 'extslt_hg', 'ring_curr']) # 
    else:
        meta_ = np.array(['cryo_x', 'cryo_y', 'cryo_z', 'cryo_t', 'pgm_en','oc_twoth','epu1_gap_readback',
                          'stemp_temp_B_T','stemp_temp_A_T', 'epu1_phase_readback', 'extslt_vg', 'extslt_hg', 'ring_curr']) # 
        meta = np.unique(np.hstack((meta, meta_)))        
    data = {}
    for i,n in enumerate(scan):
        data['six-' + str(n)] = rixs_data(n, meta=meta)
        ################################################
        # Correct the pixel shift due to different incident photon energies
        data['six-' + str(n)]['E_cali'] = E_cali
        if i==0:
            energy_ref = data['six-' + str(n)]['energy']
        energy_offset = (data['six-' + str(n)]['energy'] - energy_ref)*1000 # in the unit of meV
        if np.abs(energy_offset)>100.0:
            data['six-' + str(n)]['sig_y']+=(energy_offset/E_cali)
        else:
            pass
    #################################################################### 
    return data


##############################################################################################
def rixs_data(scan, sig_x='x_eta', sig_y='y_eta', meta=None):
    """This is for extracting the RIXS data from BNL online drive """
    """The result is a dict containing the 1-dimensional signal and interesting beamline parameters"""
    # sig_x and sig_y characterize the signal type (COM or corrected COM) provided by (Xcam) detector.
    # They can be x or x_eta, y or y_eta.
    
    # meta is a array to initialize the beamline parameters for this scan    
    
    header = db[int(scan)]
    data = {}

    # Initialize the keys for meta data
    if meta is None:
        meta = np.array(['cryo_x', 'cryo_y', 'cryo_z', 'cryo_t', 'pgm_en','pgm_cff','oc_twoth','epu1_gap_readback',
                         'stemp_temp_B_T','stemp_temp_A_T', 'epu1_phase_readback', 'extslt_vg', 'extslt_hg', 'ring_curr']) # 
    else:
        meta_ = np.array(['cryo_x', 'cryo_y', 'cryo_z', 'cryo_t', 'pgm_en','oc_twoth', 'epu1_gap_readback',
                          'stemp_temp_B_T','stemp_temp_A_T', 'epu1_phase_readback', 'extslt_vg', 'extslt_hg', 'ring_curr']) # 
        meta = np.unique(np.hstack((meta, meta_)))
    meta_name = {'cryo_x': 'x', 'cryo_y': 'y', 'cryo_z': 'z', 'cryo_t': 'th', 'pgm_en': 'energy',
                 'stemp_temp_B_T': 'T','stemp_temp_A_T':'T_cryo',
                 'epu1_phase_readback': 'pol','epu1_gap_readback': 'pol_gap','pgm_cff':'cff','oc_twoth': 'tth', 
                 'extslt_vg': 'slit_v', 'extslt_hg': 'slit_h', 'ring_curr': 'ring_curr'} # 

    #######################################################################
    # get the image dimensions
    img_size_x = int(header.config_data('rixscam')['primary'][0].get('rixscam_sensor_region_xsize')*2)
    img_size_y = int(header.config_data('rixscam')['primary'][0].get('rixscam_sensor_region_ysize'))
    data['image_size'] = np.array([img_size_x,img_size_y])
    
    #######################################################################
    # Extract the 1-D signal
    centroids = list(header.data('rixscam_centroids'))
    # pdframes = [centroid for centroid in centroids[0]]
    pdframes = [sub_centroid for centroid in centroids for sub_centroid in centroid]
    data_sheet = pd_concat([frame for frame in pdframes], ignore_index=True)  # sig in data is a pandas dataframe!

    data['sig_x'] = data_sheet[sig_x].to_numpy()
    data['sig_y'] = data_sheet[sig_y].to_numpy()

    #######################################################################
    # shift the signal from right sensor by -26 pixels
    data['sig_y'][data['sig_x']>(img_size_x/2)]-=26
    #######################################################################
    meta_list = []
    for i in db.get_table(header, stream_name="baseline").columns:
        meta_list.append(i)
        
    # Extract the meta-data
    for key in (meta):
        if key in meta_list:
            vari = db.get_table(header, stream_name="baseline", fields=[key])[key]
            if key in meta_name.keys():
                if key == 'epu1_phase_readback':
                    pol_val = vari.mean(axis=0)
                    if np.abs(pol_val - 0) < (1e-2):
                        data[meta_name[key]] = 'LH'
                    elif np.abs(pol_val - 28.5) < (1e-2):
                        data[meta_name[key]] = 'LV'
                    else:
                        data[meta_name[key]] = vari.mean(axis=0)
                else:
                    data[meta_name[key]] = vari.mean(axis=0)
            else:
                data[key] = vari.mean(axis=0)
        else:
            if key in meta_name.keys():
                data[meta_name[key]] = 'N/A'
            else:
                data[key] = 'N/A'

    # Specific the counting time for this scan
    data['image_time'] = header.config_data('rixscam')['primary'][0].get('rixscam_cam_acquire_time')  # expose time (seconds) for one image
    data['image_num'] = len(pdframes)  # Number of images in this scan
    data['count_time'] = data['image_time'] * data['image_num']  # Total counting time in seconds
    data['total_time'] = (header['stop'].get('time', time()) - header['start']['time'])  # Duration time (s) for the whole scan

    # Get the I0 for normalization
    try:
        I_0 = (db.get_table(header)['sclr_channels_chan8']).to_numpy()
        data['norm_I0'] = np.delete(I_0, np.where(I_0 <= ((I_0.max() + I_0.mean()) / 2))).mean()
    except:
        data['norm_I0'] = 1.0
    data['norm_I0']/=data['image_time']
        
    scan_start_time = ctime(header['start']['time'])

    # Print out the details for each scan
    print('--- six-{} --- points {} --- '.format(scan, data['image_num']), end='')
    print('split time {}s --- total {}s '.format(data['image_time'], data['count_time']), end='')
    print('--- duration {:.1f}s --- when {}'.format(data['total_time'], scan_start_time))

    return data


def print_data(data, scan=None):
    data_table = PrettyTable()
    data_table.field_names = ["Scan Num.", "Theta (deg)", "Inc. Energy (eV)", "2Theta (deg)",
                              "Polarization (phase)", "Exit Slit (um)", "Temp. (K)"] # 

    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))
        scan = np.sort(scan)

    for n in scan:
        data_table.add_row([n, np.round(data['six-' + str(n)]['th'], 1),
                            np.round(data['six-' + str(n)]['energy'], 1),data['six-' + str(n)]['tth'],
                            data['six-' + str(n)]['pol'], np.round(data['six-' + str(n)]['slit_v'], 1),
                            np.round(data['six-' + str(n)]['T'], 1)])
    print(data_table)


##############################################################################################
def scan_info(data_folder, scan, disp=None, sample=None):
    if sample is None:
        sample = 'six'

    data_table = PrettyTable()

    scan = np.sort(scan)

    if disp is None:
        disp_list = ['scan', 'scan_num', 'count_time', 'x', 'y', 'z', 'th', 'tth', 'energy', 'pol', 'T', 'slit_v']
    else:
        disp_list = ['scan', 'scan_num', 'count_time', 'th', 'tth', 'energy', 'pol', 'T', 'slit_v']
        for t in disp:
            if t in disp_list:
                pass
            else:
                disp_list.append(t)

    disp_data = {}

    data_table.field_names = disp_list
    for n in scan:
        try:
            data_file = h5_file(globf(data_folder + sample + '-' + str(n) + '*.hdf')[0], 'r')
        except:
            data_file = h5_file(data_folder + sample + '-' + str(n) + '.hdf', 'r')

        disp_data['scan_num'] = '{:.0f}'.format(len(np.ravel(data_file['meta']['scan'][:])))

        for disp_item in disp_list:
            if disp_item in data_file['meta'].keys():
                local_data = np.ravel(data_file['meta'][disp_item][:])
                if disp_item == 'scan':
                    if np.size(local_data) > 1:
                        disp_data[disp_item] = sample + '-' + str(local_data[0]) + '-' + str(local_data[-1])
                    else:
                        disp_data[disp_item] = sample + '-' + '{:.0f}'.format(local_data[0])
                elif disp_item == 'count_time':
                    disp_data[disp_item] = '{:.1f}/mins'.format(np.sum(local_data) / 60.)
                elif np.size(local_data) > 1:
                    disp_data[disp_item] = '{:.2f}'.format(np.mean(local_data))

                else:
                    if type(local_data[0]) == bytes:
                        disp_data[disp_item] = local_data[0].decode('ASCII')
                    else:
                        disp_data[disp_item] = '{:.2f}'.format(np.mean(local_data[0]))
            elif disp_item == 'scan_num':
                pass
            else:
                disp_data[disp_item] = '*'

        data_table.add_row(list([disp_data[i] for i in disp_list]))

    print(data_table)





def scan_data(scan,meta=None):
    """This is for extracting the data for a motor scan from BNL online drive """
    """The result is a dict containing the 1 or 2 -dimensional signal and interesting beamline parameters"""
    #############################################################
    run = db[int(scan)]
    #############################################################
    # Get the motor value
    if run.start['plan_name']=='count':
        print('This is a RIXS count, not a motor scan!!!')
        
        pass 
        
    else:
        # Initialize a dict to store the data
        data_dict = {}
        data_dict['meta'] = {}
        data_dict['data'] = {}
        #######################################################################
        # Preload expensive primary stream info from run
        primary_table = run.table(stream_name="primary")
        primary_keys = primary_table.columns

        #############################################################
        for motor in run.start['motors']:
            data_dict['data'][motor] = primary_table[motor+'_user_setpoint'].to_numpy()

        #############################################################
        # Get the data
        for detector in run.start['detectors']:
            if detector!='rixscam':
                for i in primary_keys:
                    if i[:len(detector)]==detector:
                        data_dict['data'][i] = (primary_table[i].to_numpy()).astype(float) 
            else:
                data_dict['data'][detector] = (primary_table[detector+'_xip_count_possible_event'].to_numpy()).astype(float) 

        #############################################################
        # Get the meta data

        # Initialize the keys for meta data
        if meta is None:
            meta = np.array(['cryo_x', 'cryo_y', 'cryo_z', 'cryo_t', 'pgm_en','dc_twoth',
                             'stemp_temp_B_T','stemp_temp_A_T', 'epu1_phase_readback', 'extslt_vg', 'extslt_hg', 'ring_curr']) # 
        else:
            meta_ = np.array(['cryo_x', 'cryo_y', 'cryo_z', 'cryo_t', 'pgm_en','dc_twoth',
                              'stemp_temp_B_T','stemp_temp_A_T', 'epu1_phase_readback', 'extslt_vg', 'extslt_hg', 'ring_curr']) #  
            meta = np.unique(np.hstack((meta, meta_)))

        meta_name = {'cryo_t': 'th', 'pgm_en': 'energy',
                     'stemp_temp_B_T': 'T','stemp_temp_A_T':'T_cryo', 'epu1_phase_readback': 'pol','dc_twoth': 'tth', 
                     'extslt_vg': 'slit_v', 'extslt_hg': 'slit_h', 'ring_curr': 'ring_curr'} # 'cryo_x': 'x', 'cryo_y': 'y', 'cryo_z': 'z',

        #######################################################################
        # Preload expensive baseline info from run
        baseline_table = run.table(stream_name="baseline")
        baseline_keys = baseline_table.columns

        #######################################################################
        for meta_key in (meta):
            if meta_key in run.start['motors']:
                pass
            else:
                if meta_key in baseline_keys:
                    meta_data = baseline_table[meta_key]
                    if meta_key in meta_name.keys():
                        if meta_key == 'epu1_phase_readback':
                            pol_val = meta_data.mean(axis=0)
                            if np.abs(pol_val - 0) < (1e-2):
                                data_dict['meta'][meta_name[meta_key]] = 'LH'
                            else:
                                data_dict['meta'][meta_name[meta_key]] = 'LV'
                        else:
                            data_dict['meta'][meta_name[meta_key]] = meta_data.mean(axis=0)
                    else:
                        data_dict['meta'][meta_key] = meta_data.mean(axis=0)
                else:
                    if meta_key in meta_name.keys():
                        data_dict['meta'][meta_name[meta_key]] = 'N/A'
                    else:
                        data_dict['meta'][meta_key] = 'N/A'
        return data_dict
    
    
    
    
##############################################################################################
def save_scan(data, save_folder,data_format='hdf',scan=None, sample=None):
    
    # Note: scan is a number, sample is a string!
    
    # Check the existence of save_folder, if not, create one
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    for key, data_dict in data.items():
        
        # Check the prefix for data name
        if sample is None:
            data_name = key
        else:
            data_name = sample + key[3:]
        #################################################################################################### 
        if scan is None: # Save all data if scan is not specified.
        
            ##################################################
            # Save the data
            if data_format == 'hdf':
                # Check whether the file is existing or not, if yes, delete the file
                if os.path.exists(save_folder + data_name + '.hdf') == True:
                    os.remove(save_folder + data_name + '.hdf')

                mf = h5_file(save_folder + data_name + '.hdf', 'w')
                gp1 = mf.create_group('data')
                # gp1_1 = mf.create_group('data/norm')
                gp2 = mf.create_group('meta')
                ##################################################
                # Result without normalized signal!
                for sig_key, sig in data_dict['data'].items():
                    if sig_key[-4:]!='norm':
                        if type(sig) is str:
                            t = gp1.create_dataset(sig_key, shape=(1,), dtype=special_dtype(vlen=str))
                            t[0] = sig
                        else:   
                            if type(sig) is np.ndarray:
                                gp1.create_dataset(sig_key, shape=(len(sig), 1), data=sig, dtype='float64')
                            else:
                                sig = np.array([sig]).ravel()
                                gp1.create_dataset(sig_key, shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        pass

                # Meta data
                for meta_key,meta_sig in data_dict['meta'].items():
                    if type(meta_sig) is str:
                        t = gp2.create_dataset(meta_key, shape=(1,), dtype=special_dtype(vlen=str))
                        t[0] = meta_sig
                    else:   
                        if type(meta_sig) is np.ndarray:
                            gp2.create_dataset(meta_key, shape=(len(meta_sig), 1), data=meta_sig, dtype='float64')
                        else:
                            meta_sig = np.array([meta_sig]).ravel()
                            gp2.create_dataset(meta_key, shape=(len(meta_sig), 1), data=meta_sig, dtype='float64')                                
                ##################################################                
                mf.close()
            ####################################################################################################
            elif data_format == 'txt':
                if os.path.exists(save_folder + data_name + '.txt') == True:
                    os.remove(save_folder + data_name + '.txt')

                ##################################################
                data_dict_order = {sig_key:sig for sig_key, sig in data_dict['data'].items() if sig_key[-4:]!= 'norm'} # Remove the normalized signal
                data_dict_order.update(data_dict['meta']) # Add the meta data

                ##################################################
                mdf = pd_df.from_dict(data_dict_order) # Change the dict to a Panda dataframe

                with open(save_folder + data_name + '.txt', 'w+') as f:
                    f.write(mdf.to_string(header=True, index=False))

            else:
                print('Data format is unknown!!!')
                
        else: # Only save the data specified by scan
            if key == 'six-' + str(scan):
                ##################################################
                # Save the data
                if data_format == 'hdf':
                    # Check whether the file is existing or not, if yes, delete the file
                    if os.path.exists(save_folder + data_name + '.hdf') == True:
                        os.remove(save_folder + data_name + '.hdf')

                    mf = h5_file(save_folder + data_name + '.hdf', 'w')
                    gp1 = mf.create_group('data')
                    # gp1_1 = mf.create_group('data/norm')
                    gp2 = mf.create_group('meta')
                    ##################################################
                    # Result without normalized signal!
                    for sig_key, sig in data_dict['data'].items():
                        if sig_key[-4:]!='norm':
                            if type(sig) is str:
                                t = gp1.create_dataset(sig_key, shape=(1,), dtype=special_dtype(vlen=str))
                                t[0] = sig
                            else:   
                                if type(sig) is np.ndarray:
                                    gp1.create_dataset(sig_key, shape=(len(sig), 1), data=sig, dtype='float64')
                                else:
                                    sig = np.array([sig]).ravel()
                                    gp1.create_dataset(sig_key, shape=(len(sig), 1), data=sig, dtype='float64')
                        else:
                            pass

                    # Meta data
                    for meta_key,meta_sig in data_dict['meta'].items():
                        if type(meta_sig) is str:
                            t = gp2.create_dataset(meta_key, shape=(1,), dtype=special_dtype(vlen=str))
                            t[0] = meta_sig
                        else:   
                            if type(meta_sig) is np.ndarray:
                                gp2.create_dataset(meta_key, shape=(len(meta_sig), 1), data=meta_sig, dtype='float64')
                            else:
                                meta_sig = np.array([meta_sig]).ravel()
                                gp2.create_dataset(meta_key, shape=(len(meta_sig), 1), data=meta_sig, dtype='float64')                                
                    ##################################################                
                    mf.close()
                ####################################################################################################
                elif data_format == 'txt':
                    if os.path.exists(save_folder + data_name + '.txt') == True:
                        os.remove(save_folder + data_name + '.txt')

                    ##################################################
                    data_dict_order = {sig_key:sig for sig_key, sig in data_dict['data'].items() if sig_key[-4:]!= 'norm'} # Remove the normalized signal
                    data_dict_order.update(data_dict['meta']) # Add the meta data

                    ##################################################
                    mdf = pd_df.from_dict(data_dict_order) # Change the dict to a Panda dataframe

                    with open(save_folder + data_name + '.txt', 'w+') as f:
                        f.write(mdf.to_string(header=True, index=False))

                else:
                    print('Data format is unknown!!!')
            else:
                pass
            
            
            
            
            
            
            
            
##############################################################################################
def save_map(data, save_folder, data_format='hdf'):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    data_name = 'spec-' + str(int(data['scan'].min())) + '_' + str(int(data['scan'].max()))

    if data_format == 'hdf':

        if os.path.exists(save_folder + data_name + '.hdf') == True:
            os.remove(save_folder + data_name + '.hdf')

        mf = h5_file(save_folder + data_name + '.hdf', 'w')
        gp1 = mf.create_group('map')
        gp2 = mf.create_group('cut')

        for key,sig in data.items():
            if key == 'spec':
                gp1.create_dataset(key, data=sig.T, dtype='float64')
            elif key in ['spec_x','spec_y','spec_y_shift']:
                if type(sig) is np.ndarray:
                    gp1.create_dataset(key[5:], shape=(len(sig), 1), data=sig, dtype='float64')
                else:
                    sig = np.array([sig]).ravel()
                    gp1.create_dataset(key[5:], shape=(len(sig), 1), data=sig, dtype='float64')

            elif key in ['cut_x','cut_y','cut_type','cut_interp_num','cut_wid']:

                if key in ['cut_type','cut_wid']:
                    t = gp2.create_dataset(key[4:], shape=(1,), dtype=special_dtype(vlen=str))
                    t[0] = sig
                else:
                    if type(sig) is np.ndarray:
                        gp2.create_dataset(key[4:], shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        sig = np.array([sig]).ravel()
                        gp2.create_dataset(key[4:], shape=(len(sig), 1), data=sig, dtype='float64')
            elif key=='scan':
                mf.create_dataset(key, shape=(len(sig), 1), data=sig, dtype=np.int64)
            else:
                t = mf.create_dataset(key, shape=(1,), dtype=special_dtype(vlen=str))
                t[0] = sig
        mf.close()


    elif data_format=='txt':

        print('Note: The RIXS 2D Map is not saved into the .txt file!!!')

        if os.path.exists(save_folder + data_name + '.txt') == True:
            os.remove(save_folder + data_name + '.txt')

        order_key = list(['cut_x','cut_y','cut_interp_num','cut_wid','vari','cut_type','data_channel'])

        save_dict = {}
        for key in order_key:
            save_dict[key] = data[key]

        mdf = pd_df.from_dict(save_dict)

        with open(save_folder + data_name + '.txt', 'w+') as f:
            f.write(mdf.to_string(header=True, index=False))

    else:
        print('Data format is unknown!!!')











##############################################################################################
def sig2spec(data, scan=None,
             slope_l=None, slope_r=None, points_per_pixel=None):
    if slope_l is None:
        slope_l = 0
    if slope_r is None:
        slope_r = 0
    if points_per_pixel is None:
        points_per_pixel = 3
    ##############################################################
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))

    # specify the border line between left and right sensors!
    border = data['six-' + str(scan[0])]['image_size'][0]/2
    # specify vertical size of image!
    img_row = data['six-' + str(scan[0])]['image_size'][1]
        
    # Extract the RIXS spectra from signals!
    for n in scan:
        sig_dict = data['six-' + str(n)]
        ##################################################################
        
        # Slope correction for both of sensors
        sig_l = sig_dict['sig_y'][sig_dict['sig_x'] <= border] - (
            sig_dict['sig_x'][sig_dict['sig_x'] <= border] * slope_l)  # Left sensor
        sig_r = sig_dict['sig_y'][sig_dict['sig_x'] > border] - (
            sig_dict['sig_x'][sig_dict['sig_x'] > border] * slope_r)  # Right sensor

        sig_l_x = sig_dict['sig_x'][sig_dict['sig_x'] <= border]
        sig_r_x = sig_dict['sig_x'][sig_dict['sig_x'] > border]
        
        sig_x_slope_correct = np.hstack((sig_l_x,sig_r_x))
        sig_slope_correct = np.hstack((sig_l,sig_r))
        
        # print(type(points_per_pixel))
        inten_l, edge_l = np.histogram(sig_l, bins=np.linspace(0, img_row, int(img_row * points_per_pixel) + 1))
        sig_dict['spec_x_l'] = (edge_l[:-1] + edge_l[1:]) / 2
        sig_dict['spec_y_l'] = (inten_l * points_per_pixel)

        inten_r, edge_r = np.histogram(sig_r, bins=np.linspace(0, img_row, int(img_row * points_per_pixel) + 1))
        sig_dict['spec_x_r'] = (edge_r[:-1] + edge_r[1:]) / 2
        sig_dict['spec_y_r'] = (inten_r * points_per_pixel)

        sig_dict.update(
            {'slope_l': slope_l, 'slope_r': slope_r, 'points_per_pixel': points_per_pixel, 'border': border,
            'sig_x_slope':sig_x_slope_correct,'sig_y_slope':sig_slope_correct})
        
    return data


##############################################################################################
def meta_data(data, scan=None):
    ##############################################################
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))

    meta_dict = {}

    for i, n in enumerate(scan):
        data_dict = data['six-' + str(n)]
        for key, item in data_dict.items():
            if (key not in ['sig_x', 'sig_y', 'E_cali', 'shift_type', 'rixs', 'E']) and (key[-2:] not in ['_l', '_r']):
                if key in ['image_size']:
                    if i==0:
                        meta_dict[key] = item
                    else:
                        pass
                else:
                    if i == 0:
                        meta_dict[key] = np.array([])
                    meta_dict[key] = np.append(meta_dict[key], item)
            elif key in ['slope_l', 'slope_r', 'cor_shift_l', 'cor_shift_r']:
                if i == 0:
                    meta_dict[key] = np.array([])
                meta_dict[key] = np.append(meta_dict[key], item)
            else:
                pass

    return meta_dict


##############################################################################################
def save_data(data, save_folder, scan=None,data_format='hdf', sample=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ##############################################################
    # Sort the scan in order
    if scan is None:
        scan = np.array([], dtype=int)
        for n in data.keys():
            scan = np.append(scan, int(n[4:]))
        
    for n in scan:
        if sample is None:
            data_name = 'six-' + str(int(n))
        else:
            data_name = sample + '-' + str(int(n))
            
        if data_format == 'hdf':
            if os.path.exists(save_folder + data_name + '.hdf') == True:
                os.remove(save_folder + data_name + '.hdf')
            mf = h5_file(save_folder + data_name + '.hdf', 'w')
            gp1 = mf.create_group('RawSig')
            gp1_2 = mf.create_group('data')
            gp2 = mf.create_group('data/left')
            gp3 = mf.create_group('data/right')
            gp4 = mf.create_group('meta')

            for local_key, sig in data['six-'+str(int(n))].items():
                if local_key in ['sig_x', 'sig_y','sig_x_slope','sig_y_slope']:
                    if type(sig) is np.ndarray:
                        gp1.create_dataset(local_key, shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        sig = np.array([sig]).ravel()
                        gp1.create_dataset(local_key, shape=(len(sig), 1), data=sig, dtype='float64')
                elif local_key in ['rixs', 'E']:
                    if type(sig) is np.ndarray:
                        gp1_2.create_dataset(local_key, shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        sig = np.array([sig]).ravel()
                        gp1_2.create_dataset(local_key, shape=(len(sig), 1), data=sig, dtype='float64')
                elif local_key[-2:] == '_l':
                    if type(sig) is np.ndarray:
                        gp2.create_dataset(local_key[:-2], shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        sig = np.array([sig]).ravel()
                        gp2.create_dataset(local_key[:-2], shape=(len(sig), 1), data=sig, dtype='float64')
                elif local_key[-2:] == '_r':
                    if type(sig) is np.ndarray:
                        gp3.create_dataset(local_key[:-2], shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        sig = np.array([sig]).ravel()
                        gp3.create_dataset(local_key[:-2], shape=(len(sig), 1), data=sig, dtype='float64')
                elif local_key in ['cor_shift_ref','scan']:
                    gp4.create_dataset(local_key, shape=(1, 1), data=sig, dtype=np.int64)
                    
                elif type(sig) is str:                
                    t = gp4.create_dataset(local_key, shape=(1,), dtype=special_dtype(vlen=str))
                    t[0] = sig
                    
                else:
                    if type(sig) is np.ndarray:
                        gp4.create_dataset(local_key, shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        sig = np.array([sig]).ravel()
                        gp4.create_dataset(local_key, shape=(len(sig), 1), data=sig, dtype='float64')
            mf.close()


        elif data_format == 'txt':
            if os.path.exists(save_folder + data_name + '.txt') == True:
                os.remove(save_folder + data_name + '.txt')

            order_key = list(['E', 'rixs', 'E_l', 'rixs_l', 'E_r', 'rixs_r',
                              'energy', 'th', 'tth', 'T', 'pol', 'points_per_pixel', 'slope_l', 'slope_r', 'border',
                              'E_cali'])
            save_dict = {}
            for local_key, local_sig in data['six-'+str(int(n))].items():
                if local_key in ['E', 'rixs', 'E_l', 'rixs_l', 'E_r', 'rixs_r']:
                    save_dict[local_key] = local_sig
                elif local_key in ['energy', 'th', 'tth', 'T', 'pol', 'points_per_pixel', 'slope_l', 'slope_r',
                                   'border', 'E_cali']:
                    if type(local_sig) is np.ndarray:
                        save_dict[local_key] = local_sig[0]
                    else:
                        save_dict[local_key] = local_sig
                else:
                    pass
            save_dict_order = {k: save_dict[k] for k in order_key}

            mdf = pd_df.from_dict(save_dict_order)

            with open(save_folder + data_name + '.txt', 'w+') as f:
                f.write(mdf.to_string(header=True, index=False))
        else:
            print('Data format is unknown!!!')


##############################################################################################


def save_data_total(data, save_folder, data_format='hdf', sample=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if type(data['scan']) is np.ndarray:
        data['scan'] = data['scan'].astype(int)
    else:
        data['scan'] = np.asarray(data['scan'], dtype=int)

    # points_per_pixel = int(data['points_per_pixel'][0])
    if len(data['scan']) == 1:
        if sample is None:
            data_name = 'six-' + str(data['scan'][0])
        else:
            data_name = sample + '-' + str(data['scan'][0])
    else:
        if sample is None:
            data_name = 'six-' + str(data['scan'].min()) + '_' + str(data['scan'].max())
        else:
            data_name = sample + '-' + str(data['scan'].min()) + '_' + str(data['scan'].max())

    if data_format == 'hdf':
        if os.path.exists(save_folder + data_name + '.hdf') == True:
            os.remove(save_folder + data_name + '.hdf')

        mf_total = h5_file(save_folder + data_name + '.hdf', 'w')
        gp1 = mf_total.create_group('data')
        gp11 = mf_total.create_group('data/sig')
        gp2 = mf_total.create_group('meta')

        for key, sig in data.items():
            if (key not in ['E', 'rixs']) and (key[-2:] not in ['_l', '_r', 'mb']):
                if type(sig) is np.ndarray:
                    if key in ['scan', 'cor_shift_ref']:
                        gp2.create_dataset(key, shape=(len(sig), 1), data=sig, dtype=np.int64)
                        
                    elif type(sig[0]) is np.str_:
                        t = gp2.create_dataset(key, shape=(len(sig),), dtype=special_dtype(vlen=str))
                        for tt in range(len(sig)):
                            t[tt] = sig[tt]                    
                    else:
                        gp2.create_dataset(key, shape=(len(sig), 1), data=sig, dtype='float64')                       
                elif type(sig) is str:
                    t = gp2.create_dataset(key, shape=(1,), dtype=special_dtype(vlen=str))
                    t[0] = sig[0]
                else:
                    sig = np.array([sig])
                    gp2.create_dataset(key, shape=(len(sig), 1), data=sig, dtype='float64')
            elif key in ['slope_l', 'slope_r', 'cor_shift_l', 'cor_shift_r']:
                if type(sig) is np.ndarray:
                    gp2.create_dataset(key, shape=(len(sig), 1), data=sig, dtype='float64')
                else:
                    sig = np.array([sig]).ravel()
                    gp2.create_dataset(key, shape=(len(sig), 1), data=sig, dtype='float64')
            else:
                if key in ['E', 'rixs', 'E_l', 'rixs_l', 'E_r', 'rixs_r']:
                    if type(sig) is np.ndarray:
                        gp1.create_dataset(key, shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        sig = np.array([sig]).ravel()
                        gp1.create_dataset(key, shape=(len(sig), 1), data=sig, dtype='float64')
                else:
                    if type(sig) is np.ndarray:
                        gp11.create_dataset(key, shape=(len(sig), 1), data=sig, dtype='float64')
                    else:
                        sig = np.array([sig]).ravel()
                        gp11.create_dataset(key, shape=(len(sig), 1), data=sig, dtype='float64')
        mf_total.close()

    elif data_format == 'txt':
        if os.path.exists(save_folder + data_name + '.txt') == True:
            os.remove(save_folder + data_name + '.txt')

        order_key = list(['E', 'rixs', 'E_l', 'rixs_l', 'E_r', 'rixs_r',
                          'energy', 'th', 'tth', 'T', 'pol', 'points_per_pixel', 'slope_l', 'slope_r', 'border',
                          'E_cali'])
        save_dict = {}
        for key, sig in data.items():
            if key in ['E', 'rixs', 'E_l', 'rixs_l', 'E_r', 'rixs_r']:
                save_dict[key] = sig
            elif key in ['energy', 'th', 'tth', 'T', 'pol', 'points_per_pixel', 'slope_l', 'slope_r', 'border',
                         'E_cali']:
                if type(sig) is np.ndarray:
                    save_dict[key] = sig[0]
                else:
                    save_dict[key] = sig
            else:
                pass

        save_dict_order = {k: save_dict[k] for k in order_key}

        mdf = pd_df.from_dict(save_dict_order)

        with open(save_folder + data_name + '.txt', 'w+') as f:
            f.write(mdf.to_string(header=True, index=False))
    else:
        print('Data format is unknown!!!')


##############################################################################################
def correlate_shift(sig_x_ref, sig_ref, sig_x, sig):
    if len(sig) != len(sig_ref):
        raise ValueError("The lengths of signals for correlations are different!!!")

    if (len(np.unique(np.diff(sig_x))) != 1) or (len(np.unique(np.diff(sig_x_ref))) != 1):
        sig_x = np.linspace(sig_x.min(), sig_x.max(), len(sig_x))
        sig_x_ref = np.linspace(sig_x_ref.min(), sig_x_ref.max(), len(sig_x_ref))

    if (sig_x == sig_x_ref).all() != True:
        sig = np.interp(sig_x_ref, sig_x, sig)
        sig_x = sig_x_ref
    else:
        pass

    sig_x_ref = sig_x_ref.astype(float)
    sig_ref = sig_ref.astype(float)
    sig_x = sig_x.astype(float)
    sig = sig.astype(float)
    sig_size = sig.size

    sig_ref = np.subtract(sig_ref, sig_ref.mean(), out=sig_ref, casting='unsafe')
    sig_ref = np.divide(sig_ref, sig_ref.std(), out=sig_ref, casting='unsafe')
    sig = np.subtract(sig, sig.mean(), out=sig, casting='unsafe')
    sig = np.divide(sig, sig.std(), out=sig, casting='unsafe')

    cross_correlation = correlate(sig, sig_ref)

    dt = np.arange(1 - sig_size, sig_size)

    recovered_shift = -dt[np.argmax(cross_correlation)]

    sig_x_step = np.unique(np.diff(sig_x))[0]

    xshift = (recovered_shift) * sig_x_step

    return xshift


##############################################################################################
def sig_cor(x_ref, y_ref, x, y, roi_cor=None, cor_interp=1):
    if roi_cor is None:
        roi_cor = [x_ref.min(), x_ref.max()]

    roi_ref = np.where((x_ref >= roi_cor[0]) & (x_ref <= roi_cor[1]))[0]
    roi_ref_len = len(roi_ref)

    roi = np.where((x >= roi_cor[0]) & (x <= roi_cor[1]))[0]

    # Make sure the signals for correlation have same length.
    x_cor = x[roi.min():(roi.min() + roi_ref_len)]
    y_cor = y[roi.min():(roi.min() + roi_ref_len)]
    x_ref_cor = x_ref[roi_ref.min():(roi_ref.min() + roi_ref_len)]
    y_ref_cor = y_ref[roi_ref.min():(roi_ref.min() + roi_ref_len)]

    x_cor_itp, y_cor_itp = m_interpolate(x_cor, y_cor, len(x_cor) * int(cor_interp))
    x_ref_cor_itp, y_ref_cor_itp = m_interpolate(x_ref_cor, y_ref_cor, len(x_ref_cor) * int(cor_interp))

    shift = correlate_shift(x_ref_cor_itp, y_ref_cor_itp, x_cor_itp, y_cor_itp)

    return shift

##############################################################################################

##############################################################################################
# Smooth function
def savgol(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def smooth(y, wid=1, polyorder=1, deriv=0, rate=1, times=1):
    sig = y
    if (wid % 2) == 0:
        wid += 1
    for t in range(times):
        sig = savgol(y, wid, polyorder, deriv=0, rate=1)
        y = sig
    return sig


def m_interpolate(sig_x, sig_y, size, order=3):
    # Do the interpolation
    # sig_x must be sorted
    sig_y_sort = sig_y[sig_x.argsort()]
    sig_x_sort = np.sort(sig_x)

    tck = interpolate.splrep(sig_x_sort, sig_y_sort, w=np.ones(len(sig_x_sort)), s=0, k=order)
    xnew = np.linspace(sig_x_sort.min(), sig_x_sort.max(), size)
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew

#########################################################################################################################
def get_para(fit_params):
    # From the fitting parameters class of lmfit to get the fitting results
    # A dictionary has been returned
    para_r = np.array([])
    for d_key in (fit_params.valuesdict()).keys():
        loc_r_list = [(fit_params.valuesdict())[d_key], (fit_params[d_key]).stderr]
        para_r = np.append(para_r, loc_r_list, axis=0)
    para_r = para_r.reshape(int(len(para_r) / 2), 2)
    para_dict = dict(zip((fit_params.valuesdict()).keys(), para_r))
    return para_dict

########################################################################
def save_dict2hdf(data_folder,data_dict):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if os.path.exists(data_folder + data_dict['data_file'] + '.hdf') == True:
        os.remove(data_folder + data_dict['data_file'] + '.hdf')
    
    mf = h5_file(data_folder + data_dict['data_file'] + '.hdf', 'w')
    for local_key, sig in data_dict.items():
        
        if local_key!='data_file':
            if type(sig) is np.ndarray:
                mf.create_dataset(local_key, shape=(len(sig), 1), data=sig, dtype='float64')
            else:
                sig = np.array([sig]).ravel()
                mf.create_dataset(local_key, shape=(len(sig), 1), data=sig, dtype='float64')
        else:
            t = mf.create_dataset(local_key, shape=(1,), dtype=special_dtype(vlen=str))
            t[0] = sig
    mf.close()
        
    
    
  
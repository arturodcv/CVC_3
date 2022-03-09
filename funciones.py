import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from PIL import Image
from glob import glob
import shutil
import nest
import nest.topology as tp
import pickle
from collections import Counter
from collections import OrderedDict
import scipy
from scipy.signal import fftconvolve
from scipy.signal import find_peaks
from datetime import datetime
import sys

from nest_values import *

#Lambda = int(sys.argv[3]); Sigma = Lambda * 0.36
#max_rescaling_factor_gabor =  int(sys.argv[4])
#################################################### Folders ################################################################

def create_folder(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)

def remove_contents(path_name):
    for filename in os.listdir(path_name):
        file_path = os.path.join(path_name, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            continue

##################################################### Gabor ####################################################################

def gabor_filter(K_size,Lambda, Theta, Sigma, Gamma, Psi):
    sigma_x = Sigma
    sigma_y = float(Sigma) / Gamma
    (y, x) = np.meshgrid(np.arange(- K_size // 2,K_size // 2), np.arange(-K_size // 2,K_size // 2))
    x_ = np.cos(Theta) * x + np.sin(Theta) * y 
    y_ = -np.sin(Theta) * x + np.cos(Theta) * y
    gabor = np.exp(-0.5 * (x_**2 / sigma_x**2 + y_**2 / sigma_y**2)) * np.cos(2*np.pi*x_/Lambda + Psi)
    return gabor

def apply_filter(gray_img, K_size, Lambda, Theta, Sigma, Gamma, Psi):
    gray = np.pad(gray_img, (K_size//2, K_size//2), 'edge')
    gabor = gabor_filter(K_size = K_size, Lambda = Lambda, Theta = Theta, Sigma = Sigma, Gamma = Gamma, Psi = Psi)
    gabor[gabor < 0.0] = (gabor[gabor < 0.0])* 1.27679  
    output = fftconvolve(gray,gabor, mode = "valid")
    return output

def gabor(gray_img,orientation_in_radians):
    output = np.zeros((gray_img.shape[0],gray_img.shape[1]), dtype=np.float32) 
    orientation_ = orientation_in_radians*math.pi/180
    output = apply_filter(gray_img, K_size=K_size, Lambda=Lambda, Theta=orientation_, Sigma=Sigma, Gamma=Gamma,Psi = Psi )
    output = np.clip(output, 0, max(0,np.max(output)))
    return output

########################################################### Nest ###################################################################

def input_treatment(input_spike,x_cortex_size,y_cortex_size,orientation):
    input_as_img = Image.fromarray(input_spike)
    input_resized = np.asarray(input_as_img.resize((x_cortex_size,y_cortex_size), resample = Image.NEAREST  ))
    input_transposed = input_resized.transpose()
    input_as_list = input_transposed.tolist()
    flat_list = [item for sublist in input_as_list for item in sublist]
    return flat_list
    
def main_img(img,orientation, max_to_rescale):
    img = cv2.imread(img)
    if correct_gamma == True: 
        img_in_0_1 = np.multiply(img,1/255)
        corrected_gamma = np.power(img_in_0_1,1/2.2); 
        img_in_0_255 = np.multiply(corrected_gamma,255.0)
        format_img = np.uint8(img_in_0_255)
        img = format_img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    output_gabor = gabor(gray_img,orientation)
    
    output_gabor = np.multiply(output_gabor,1/(max_to_rescale/max_rescaling_factor_gabor))
    
    if cut_pixels != 0:
        output_gabor = output_gabor[cut_pixels:-cut_pixels,cut_pixels:-cut_pixels]
        
    flat_list = input_treatment(output_gabor,x_cortex_size,y_cortex_size,orientation)

    return flat_list
    
def full_img_filtering(images_to_simulate,num_orientations):
    gabors_dict = {}
    max_gabor = (gabor_filter(K_size = K_size, Lambda = Lambda, Theta = 90 * math.pi/ 180, Sigma = Sigma, Gamma = 0.00001, Psi = Psi) + 1) * 255 / 2
    max_gabor = np.max(gabor(max_gabor,90))

    for i in range(0,len(images_to_simulate)):
        image_dict = {}
        for j in range(0,num_orientations): 
            orientation = j*180/num_orientations
            image_dict["orientation_"+str(orientation)] = main_img(images_to_simulate[i],orientation,max_gabor)
        gabors_dict["image_"+str(i)] = image_dict
    return gabors_dict

def save_gabors(gabors_to_nest, images_to_simulate,num_orientations):
    for i in range(0,len(images_to_simulate)):
        for j in range(0, num_orientations):
            orientation = j*180/num_orientations
            img_to_save = np.asarray(gabors_to_nest["image_"+str(i)]["orientation_"+str(orientation)]).reshape((x_cortex_size,y_cortex_size)).transpose()
            if get_output_gabors:
                plt.imshow(img_to_save); plt.colorbar(); plt.title("orientation_" + str(orientation)); 
                plt.savefig(gabor_folder + "/orientation_"+str(int(orientation)) + ".png"); plt.close("all")
            else:
                plt.imsave("gabor_outputs/orientation_" + str(orientation) + '.png',img_to_save, cmap = "gray")

    
def main_create():
    l_exc  = create_layer(x_cortex_size,y_cortex_size,extent,'exc',neurons_per_column_exc)
    l_inh  = create_layer(x_cortex_size,y_cortex_size,extent,'inh',neurons_per_column_inh)
    sd_exc = nest.Create('spike_detector', params = {'to_file': True, 'to_memory': False})
    sd_inh = nest.Create('spike_detector', params = {'to_file': True, 'to_memory': False})
    l_poiss = create_layer(x_cortex_size,y_cortex_size,extent,'poisson_generator',1)
    return l_exc,l_inh,sd_exc,sd_inh,l_poiss
    
def main_one_orientation(orientation):
    orientation = orientation*math.pi/180
    l_exc,l_inh,sd_exc,sd_inh,l_poiss = main_create()
    lgn_to_v1_connections(l_exc,l_inh,sd_exc,sd_inh,l_poiss ,orientation)
    orientation = orientation*180 / math.pi
    lyrs_dict = {'l_exc_'+str(orientation): l_exc,'l_inh_'+str(orientation): l_inh,'sd_exc_'+str(orientation): sd_exc,'sd_inh_'+str(orientation): sd_inh}
    poiss_dict = {'l_poiss_'+str(orientation): l_poiss}
    return lyrs_dict, poiss_dict



def main_all_orientations(num_orientations):
    lyrs = {}
    poiss = {}
    for i in range(0,num_orientations):
        orientation =  i*180/num_orientations 
        lyrs["orientation_" + str(orientation)], poiss["orientation_"+str(orientation)] = main_one_orientation(orientation)
    for i in range(0,num_orientations):
        for j in range(0,num_orientations):
            or_i = i*180/num_orientations;
            or_j = j*180/num_orientations;
            l_exc_i = lyrs['orientation_'+str(or_i)]['l_exc_'+str(or_i)]
            l_inh_i = lyrs['orientation_'+str(or_i)]['l_inh_'+str(or_i)]
            l_exc_j = lyrs['orientation_'+str(or_j)]['l_exc_'+str(or_j)] 
            l_inh_j = lyrs['orientation_'+str(or_j)]['l_inh_'+str(or_j)]
            
            if lateral_connections:   
                v1_lateral_connections(l_exc_i,l_exc_j,l_inh_i,l_inh_j,i*math.pi/num_orientations,j*math.pi/num_orientations)
    return lyrs, poiss
 
def set_poisson_values(img_dict, poiss_layers,num_orientations):
    for i in range(0,num_orientations):
        orientation = i*180/num_orientations
        filtered_img =  img_dict["orientation_"+str(orientation)]

        fixed_list = [k * factor if k > 10.0 else (11 - k)**0.1 * k * factor for k in filtered_img]; 
        fixed_list = [fixed_list[k] if filtered_img[k] > poisson_bias else poisson_bias * factor_bias  for k in range(0,len(fixed_list))]

        l_poiss = list(poiss_layers['orientation_'+str(orientation)]['l_poiss_' + str(orientation)])
        nest.SetStatus(nest.GetNodes(l_poiss)[0],'rate', fixed_list)

################################################################# Layers #################################################################

def create_layer(rows,columns,extent,elements,neurons_per_column):
    return tp.CreateLayer({'rows': rows,'columns': columns,'extent': extent,'elements': [elements,neurons_per_column],'edge_wrap': edge_wrap}) 

########################################################### Connetivity ##################################################################

def create_lat_exc(kernel_type,kappa,orientation_i,orientation_j, weight_type, delay, slowness):
    return  {'connection_type': 'convergent',     
             'mask': {'circular': {'radius': radius_lat}}, 
             'kernel': {kernel_type: {'kappa': kappa,'orientation_i': orientation_i, 'orientation_j': orientation_j, 'rescale': rescale }},
             'weights': weight_type, 
             'delays': {'linear':{'c':delay,'a':slowness}}, 
             'synapse_model': syn_model_exc,
             'allow_autapses': allow_autapses, 'allow_multapses': allow_multapses
            }     
             

def lgn_to_v1_connections(l_exc,l_inh,sd_exc,sd_inh,l_poiss, self_orientation):
    tp.ConnectLayers(l_poiss, l_exc, dict_poiss_to_v1_exc)
    tp.ConnectLayers(l_poiss, l_inh, dict_poiss_to_v1_inh)
    leaves_exc = nest.GetLeaves(l_exc, local_only=True)[0]
    nest.Connect(leaves_exc, sd_exc)
    leaves_inh = nest.GetLeaves(l_inh, local_only=True)[0]
    nest.Connect(leaves_inh, sd_inh)

    
def v1_lateral_connections(l_exc_i,l_exc_j,l_inh_i,l_inh_j,orientation_i,orientation_j):
    if orientation_i == orientation_j:
        tp.ConnectLayers(l_exc_i, l_inh_j, short_range_exc_inh) 
        tp.ConnectLayers(l_exc_i, l_exc_j, short_range_exc_exc)
        
    tp.ConnectLayers(l_inh_i, l_inh_j, short_range_inh_inh) 
    tp.ConnectLayers(l_inh_i, l_exc_j, short_range_inh_exc)
    tp.ConnectLayers(l_exc_i, l_exc_j, create_lat_exc('PlosOne_J',kappa_j,orientation_i,orientation_j, weight_large_range_exc_exc, delay_exc_large, slowness_exc_large))
    tp.ConnectLayers(l_exc_i, l_inh_j, create_lat_exc('PlosOne_W',kappa_w,orientation_i,orientation_j, weight_large_range_exc_inh, delay_exc_large, slowness_exc_large))

    
############################################################## Data ###############################################################

def save_dict(to_save,name_to_save):
    to_record = open( name_to_save + ".pkl", "wb")
    pickle.dump(to_save, to_record)
    to_record.close()

def load_dict(name):
    to_load = open(name +str('.pkl'), "rb")
    data = pickle.load(to_load)
    to_load.close()
    return data
    
########################################################## Results ##################################################################

def read_and_fix_dataframe(orientation_to_read,exc_or_inh):
    data = pd.read_pickle(df_folder + '/data_' + exc_or_inh + '_' + str(orientation_to_read) + '.pkl')
    data = data.sort_values(by=['Time'])
    data = data.set_index([pd.Index([i for i in range(len(data))])])
    return data

def get_times(data):
    times = data['Time'].tolist() ; times = (np.around(data['Time'],window_time)).tolist()
    unique_times = np.unique(np.array(times)).tolist()
    full_times = [i for i in range(0,int(simulation_time + ms_steady_state_simulation) + 1)]
    complementary_time_list = list(set(full_times) - set(unique_times))
    return times,complementary_time_list

def generate_frames(data):
    create_folder(plots_path)
    remove_contents(plots_path)

    times = data['Time'].tolist() 
    times = (np.around(data['Time'],window_time)).tolist()
    actual_time = times[0]
    x_pos = ( np.around( (data['x_pos'] + extent[0] * 0.5) * 10 - 0.5)).astype(int).tolist()
    y_pos = ( np.around( (data['y_pos'] + extent[0] * 0.5) * 10 - 0.5)).astype(int).tolist()

    num_spikes = len(data)
    array = [0] * x_cortex_size * y_cortex_size
    
    for i in range(0,num_spikes):
        if times[i] - actual_time != 0 or i == num_spikes:
            name = plots_path+'/plot_time_'+str(actual_time)+'.tiff'
            array = np.reshape(array,(x_cortex_size,y_cortex_size))
            img = Image.fromarray(np.uint8(array),'L').resize(re_size).transpose(Image.FLIP_TOP_BOTTOM)
            img.save(name,compress_level = 1)
            actual_time = times[i]
            array = [0] * x_cortex_size * y_cortex_size
        array[ x_pos[i] + y_cortex_size * y_pos[i] ] += 1

    name = plots_path+'/plot_time_'+str(actual_time)+'.tiff'
    array = np.reshape(array,(x_cortex_size,y_cortex_size))
    img = Image.fromarray(np.uint8(array),'L').resize(re_size).transpose(Image.FLIP_TOP_BOTTOM)
    img.save(name,compress_level = 1)
    return times
    
def generate_empty_frames(times):
    unique_times = np.unique(np.array(times)).tolist()
    full_times = [i for i in range(0,int(simulation_time) + 1)]
    complementary_time_list = list(set(full_times) - set(unique_times))

    array = [0] * x_cortex_size * y_cortex_size
    array = np.reshape(array,(x_cortex_size,y_cortex_size))
    img = Image.fromarray(np.uint8(array),'L').resize(re_size).transpose(Image.FLIP_TOP_BOTTOM)

    for i in complementary_time_list:
        name = plots_path+'/plot_time_'+str(i)+'.tiff'
        img.save(name,compress_level = 1)
    frames = unique_times + complementary_time_list
    frames.sort()
    return frames, complementary_time_list
    
def read_frames(frames):
    img_array = []
    for milisecond in frames[:-1]:
        filename = plots_path + '/plot_time_' + str(milisecond)+'.tiff'
        img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        img_array.append(img)
    return img_array
    
def create_video(img_array,orientation_to_read ,exc_or_inh, path):
    size = (img_array[0].shape[0],img_array[0].shape[1])
    video_out = cv2.VideoWriter(path + '/neurons_video_' + str(orientation_to_read) + '_' + str(exc_or_inh)+'_.avi',0, frames_per_second, size)
    max_array = np.max(img_array)

    for img in img_array:
        img = (np.multiply(img, 255 / max_array)).astype(np.uint8)
        img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
        video_out.write(img)
    video_out.release()

    
def create_avg_img(img_array,orientation_to_read ,exc_or_inh, path ):
    img_sum = np.zeros(re_size)
    if exc_or_inh == 'l_exc':
        for img in img_array[image_from:]:
            img_sum = img_sum + np.divide(img,neurons_per_column_exc)
    if exc_or_inh == 'l_inh':
        for img in img_array[image_from:]:
            img_sum = img_sum + np.divide(img,neurons_per_column_inh)

    plt.imshow(img_sum);  plt.title('Average image'); plt.colorbar(); 
    plt.savefig(path + '/Average_img_' + str(orientation_to_read) + '_' + str(exc_or_inh)+'_.png');
    plt.close('all')

    
def get_eeg(times, complementary_time_list, orientation_to_read, exc_or_inh, path):
    eeg = Counter(times)
    eeg.update({i:0 for i in complementary_time_list})
    eeg = list(OrderedDict(sorted(eeg.items())).values())
    save_dict(eeg,path + '/eeg_' + str(orientation_to_read) + str(exc_or_inh))
    

    plt.plot(eeg); plt.title('EEG signal'); plt.xlabel("Time (ms)"); plt.ylabel("Number of spikes")
    plt.savefig(path + '/eeg_' + str(orientation_to_read) + '_' + str(exc_or_inh)+'_.png')
    plt.close('all')
    return eeg
    
def get_frequencies(eeg,orientation_to_read,exc_or_inh, path):
    #f, t, Sxx = scipy.signal.spectrogram(np.asarray(eeg), fs = len(eeg)/2)
    #plt.pcolormesh(t, f, Sxx, shading='gouraud'); plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]'); plt.grid(); 
    #plt.savefig(path + '/spectogram_' + str(orientation_to_read) + '_' + str(exc_or_inh)+'_.png')
    #plt.close('all')
    
    freqs, density = scipy.signal.periodogram(eeg[eeg_freqs_from:eeg_freqs_until],fs = 1000, scaling = 'density');
    peaks, values = find_peaks(density, height= 0.01, distance = 10); 
    idx = (- values['peak_heights']).argsort()[:num_max_frequencies]
    
    for node,peak_value in zip(peaks[idx].tolist(), values['peak_heights'][idx].tolist()):
        print("Node    Narrowband      Broadband[", broadband_initial , ',' , broadband_end_1, ']')
        print(node,'     ',np.around(peak_value,2), '     ',np.around(sum(density[broadband_initial:broadband_end_1]),2))
        print("Node    Narrowband      Broadband[", broadband_initial , ',' , broadband_end_2, ']')
        print(node,'     ',np.around(peak_value,2), '     ',np.around(sum(density[broadband_initial:broadband_end_2]),2))
    plt.plot(freqs,density); plt.xlabel("Frequency (Hz)"); plt.ylabel("Frequency Domain (Spectrum) Magnitude")
    plt.grid(); plt.savefig(path + '/periodogram_' + str(orientation_to_read) + '_' + str(exc_or_inh)+'_.png')
    plt.close('all')
    return density, peaks, idx
    
def collect_data(image_selected, exc_eeg, inh_eeg, peaks_exc, freqs_exc, idx_exc,peaks_tot,freqs_tot,idx_tot, seed):
    dictionary = {'image_name': image_selected, 'exc_activity': np.sum(exc_eeg), 'inh_activity': np.sum(inh_eeg), 
                  'exc_spikes_from': np.sum(exc_eeg[200:]), 'inh_spikes_from': np.sum(inh_eeg[200:]),
                  'node_exc': peaks_exc[idx_exc][0], 'gamma_power_exc': np.around(sum(freqs_exc[broadband_initial:broadband_end_1]),2) ,
                  'node_tot': peaks_tot[idx_tot][0], 'gamma_power_tot': np.around(sum(freqs_tot[broadband_initial:broadband_end_1]),2) ,
                  'seed': seed, 
                  'selected_hypercolumns': radius,
                  'ms_stimulus': ms_per_stimuli, 'neurons_per_column_inh': neurons_per_column_inh,
                  'mean_lat_conn_inh': mean_lat_conn_inh, 
                  'stddev_lat_conn_inh': stddev_lat_conn_inh,'mean_lat_conn_exc': mean_lat_conn_exc, 
                  'stddev_lat_conn_exc': stddev_lat_conn_exc, 'delay_exc': delay_exc,
                  'delay_exc_large': delay_exc_large, 'delay_inh': delay_inh, 
                  'slowness_exc': slowness_exc, 'slowness_exc_large': slowness_exc_large,
                  'slowness_inh': slowness_inh, 'kappa_j': kappa_j, 'kappa_w': kappa_w, 
                  'weight_large_range_exc_exc': weight_large_range_exc_exc,
                  'weight_large_range_exc_inh': weight_large_range_exc_inh,'rescale': rescale, 
                  'ratio_inh_exc_w': ratio_inh_exc_w,'p_center_inh_exc': p_center_inh_exc, 
                  'weight_inh_exc': weight_inh_exc,'p_center_inh_inh': p_center_inh_inh,
                  'weight_inh_inh': weight_inh_inh,'p_center_exc_exc': p_center_exc_exc,
                  'weight_exc_exc': weight_exc_exc,
                  'p_center_exc_inh': p_center_exc_inh,'weight_exc_inh': weight_exc_inh,
                  'input_weight_poiss_inh': input_weight_poiss_inh
                  }
    now = datetime.now()
    create_folder(collect_data_folder)
    save_dict(dictionary,collect_data_folder + '/results_' + str(image_selected) +'_'+ str(now)[:-7])
                  
def get_kurtosis(data,data_type):
    data_ = data[["x_pos","y_pos"]]
    counts = data_.groupby(["x_pos", "y_pos"]).size().reset_index(name="counts")['counts'].tolist()
    counts.sort(reverse = True)
    counts_mirror = counts[::-1] + counts  
    plt.plot([i for i in range(-len(counts),len(counts)) ], counts_mirror);  plt.title("Distribution of " + str(data_type)); 
    plt.savefig(results_path + "/kurtosis_" + str(data_type) + ".png"); plt.close("all")
    kurt = scipy.stats.kurtosis(counts_mirror)
    return kurt    
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  

import sys


weight_exc_inh = float(sys.argv[3])
weight_inh_exc = float(sys.argv[4])


#Paths
sd_path = 'spk_detectors_folder' 
df_folder = 'dataframes_folder'
plots_path = 'plots_folder' 
results_path = 'results_folder'
positions_path = 'positions_folder'
input_images_path = 'input_images_folder' 
gabor_folder = 'gabor_outputs'
collect_data_folder = 'massive_search_2/results_' + str(weight_exc_inh) + '_' + str(weight_inh_exc) 


#Simulation 
ms_per_stimuli = 700.0
simulation_time = ms_per_stimuli * 1 #num_images_to_simulate 
ms_steady_state_simulation = 0.0

lateral_connections = True

#Size
num_hipercolumns = 9
columns_in_hipercolumns = 10
x_cortex_size = num_hipercolumns * columns_in_hipercolumns 
y_cortex_size = num_hipercolumns * columns_in_hipercolumns
cortex_size = x_cortex_size * y_cortex_size

#Gabor
K_size = 900 ;Psi = 0 ; Gamma = 0.7 ; Lambda = 50 ; Sigma = Lambda * 0.36
cut_pixels = 0
get_output_gabors = 0
max_rescaling_factor_gabor = 70
correct_gamma = 0

#Nest
local_num_threads = 2
resolution = 0.1

#Number of orientations
num_orientations = 4

#Layers
extent = [float(num_hipercolumns), float(num_hipercolumns)]
ratio_exc_inh = 4
neurons_per_column_inh = 5
neurons_per_column_exc = ratio_exc_inh * neurons_per_column_inh 
poisson_bias = 4.0

#Poisson
factor = 430.9
factor_bias = 430.9

############ Dictionaries

#Layers
edge_wrap = False #True 
allow_autapses = True
allow_multapses = False
  

#Kernel
stddev_c_rf = 0.08 

p_center_inh = 1.0
mean_lat_conn_inh = 0.0 ##!!!!!!!!!!!!!!!!!!!!!!!!! 3 * stddev_c_rf
stddev_lat_conn_inh = 1 * stddev_c_rf 
input_stddev_inh = 0.0

p_center_exc = p_center_inh * 1
mean_lat_conn_exc = 0.0
stddev_lat_conn_exc = stddev_lat_conn_inh * 2
input_stddev_exc = 0.0

n_sigmas_inside_mask = 3.5
n_microcolumn_height = 10 
n_microcolumn_width  = 10
n_rows_latconn_inh   = int(stddev_lat_conn_inh * n_sigmas_inside_mask * 2 * n_microcolumn_height) + 1 
n_cols_latconn_inh   = int(stddev_lat_conn_inh * n_sigmas_inside_mask * 2 * n_microcolumn_width)  + 1 
n_rows_latconn_exc   = int(stddev_lat_conn_exc * n_sigmas_inside_mask * 2 * n_microcolumn_height) + 1
n_cols_latconn_exc   = int(stddev_lat_conn_exc * n_sigmas_inside_mask * 2 * n_microcolumn_width)  + 1  


#Synapse model
syn_model_inh = 'static_synapse_hpc'
syn_model_exc = 'static_synapse_hpc' 

#Delays
delay_exc = 1.5 ; delay_exc_large = 1.5 ; delay_inh = 0.75
slowness_exc = 0.5; slowness_exc_large = 0.5; slowness_inh = 0.25


# Large lateral connections
kappa_j = 0.126 * 1
kappa_w = 0.14 * 1
weight_large_range_exc_exc = 0.001
weight_large_range_exc_inh = 0.02
rescale = 1.0 ; radius_lat = 3.5#2.5

# Short lateral connections
ratio_inh_exc_w = 4.0
p_center_inh_exc = 1.0 ;        weight_inh_exc = - ratio_inh_exc_w * weight_inh_exc
p_center_inh_inh = 1.0 ;        weight_inh_inh = - ratio_inh_exc_w * 0.1
p_center_exc_exc = 1.0 ;        weight_exc_exc =  0.4
p_center_exc_inh = 1.0 ;        weight_exc_inh =  weight_exc_inh

input_weight_poiss_exc = 1.0     
input_weight_poiss_inh = 0.3
               
dict_poiss_to_v1_exc  = {'connection_type': 'divergent','weights': input_weight_poiss_exc ,
                         'mask': {'grid': {'rows': 1, 'columns': 1}}, 
                         'allow_autapses': allow_autapses, 'allow_multapses': allow_multapses}

dict_poiss_to_v1_inh  = {'connection_type': 'divergent','weights': input_weight_poiss_inh ,
                         'mask': {'grid': {'rows': 1, 'columns': 1}}, 
                         'allow_autapses': allow_autapses, 'allow_multapses': allow_multapses}

dict_divergent    = {'connection_type': 'divergent','mask': {'grid': {'rows': 1, 'columns': 1}},
                      'allow_autapses': allow_autapses, 'allow_multapses': allow_multapses}

short_range_inh_exc =  {'connection_type': 'convergent',
                        'mask': {'grid':{'rows':n_rows_latconn_inh ,'columns':n_cols_latconn_inh }, 
                                 'anchor':{'row':(n_rows_latconn_inh - 1)//2,'column':(n_cols_latconn_inh - 1)//2}},
                        'delays': {'linear':{'c':delay_inh,'a':slowness_inh}},
                        'kernel': {'gaussian':{'p_center': p_center_inh_exc , 'sigma':stddev_lat_conn_inh , 'mean': mean_lat_conn_inh}},
                        'weights': weight_inh_exc ,
                        'synapse_model':syn_model_inh,
                        'allow_autapses': allow_autapses, 'allow_multapses': allow_multapses}
                       
short_range_inh_inh =  {'connection_type': 'convergent',
                        'mask': {'grid':{'rows':n_rows_latconn_inh,'columns':n_cols_latconn_inh}, 
                                 'anchor':{'row':(n_rows_latconn_inh - 1)//2,'column':(n_cols_latconn_inh - 1)//2}},
                        'delays': {'linear':{'c':delay_inh,'a':slowness_inh}},
                        'kernel': {'gaussian':{'p_center': p_center_inh_inh  , 'sigma':stddev_lat_conn_inh , 'mean': mean_lat_conn_inh}}, 
                        'weights': weight_inh_inh ,
                        'synapse_model':syn_model_inh,
                        'allow_autapses': allow_autapses, 'allow_multapses': allow_multapses}
                       
short_range_exc_exc =  {'connection_type': 'convergent',
                        'mask': {'grid':{'rows':n_rows_latconn_exc,'columns':n_cols_latconn_exc}, 
                                 'anchor':{'row':(n_rows_latconn_exc - 1)//2,'column':(n_cols_latconn_exc - 1)//2}},
                        'delays': {'linear':{'c':delay_exc,'a':slowness_exc}},
                        'kernel': {'gaussian':{'p_center': p_center_exc_exc, 'sigma':stddev_lat_conn_exc , 'mean':mean_lat_conn_exc}}, 
                        'weights':  weight_exc_exc  , 
                        'synapse_model':syn_model_exc,
                        'allow_autapses': allow_autapses, 'allow_multapses': allow_multapses}
                       
short_range_exc_inh =  {'connection_type': 'convergent',
                        'mask': {'grid':{'rows':n_rows_latconn_exc,'columns':n_cols_latconn_exc}, 
                                 'anchor':{'row':(n_rows_latconn_exc - 1)//2,'column':(n_cols_latconn_exc - 1)//2}},
                        'delays': {'linear':{'c':delay_exc,'a':slowness_exc}},
                        'kernel': {'gaussian':{'p_center': p_center_exc_inh  , 'sigma':stddev_lat_conn_exc, 'mean':mean_lat_conn_exc}}, 
                        'weights':  weight_exc_inh , 
                        'synapse_model':syn_model_exc,
                        'allow_autapses': allow_autapses, 'allow_multapses': allow_multapses}
                       

#Modelos 
RS_dict =  {'a':0.02, 'b':0.2, 'c':-65.,'d':8.0, 'V_th':30.}
FS_dict =  {'a':0.1, 'b':0.2, 'c':-65., 'd':2.0, 'V_th':30.}



#Results (image, video, EEG, frequencies)
window_time = 0
re_size = (x_cortex_size,y_cortex_size)
frames_per_second = 20
num_max_frequencies = 2
broadband_initial = 10
broadband_end_1 = 50
broadband_end_2 = 60
image_from = int(ms_steady_state_simulation) + 100
eeg_freqs_from = int(ms_steady_state_simulation) + 200
eeg_freqs_until = eeg_freqs_from + 500
radius = 2.5
make_image_video = False
collect_data_bool = 0




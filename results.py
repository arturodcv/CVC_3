#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from PIL import Image
import time
from glob import glob
import shutil
from tqdm import tqdm
from collections import Counter
from collections import OrderedDict
from scipy.fft import rfft
import pickle
import scipy.signal
import sys

from nest_values import *
from funciones   import *

image_selected = sys.argv[1] 
Lambda = sys.argv[2] 

with open('seed.txt') as f:
    seed = lines = int(f.readlines()[0])

########################### results for exc/inh/total 

print("Results for total data: ")
path = results_path + '/data_total'; create_folder(path) ; remove_contents(path)   

data = read_and_fix_dataframe('','total')
times,complementary_time_list = get_times(data)
total_eeg = get_eeg(times, complementary_time_list, 'total', '_', path)
freqs, peaks, idx = get_frequencies(total_eeg,'total','_', path)
total_eeg = np.sum(total_eeg)

print("\n\nResults for inhibitory data: ")
path = results_path + '/data_inh'; create_folder(path) ; remove_contents(path)

data = read_and_fix_dataframe('','inh')
times,complementary_time_list = get_times(data)
inh_eeg = get_eeg(times, complementary_time_list, 'inh', '_', path)
print('inhibitory spikes: ',np.sum(inh_eeg[200:]))
freqs, peaks, idx = get_frequencies(inh_eeg,'inh','_', path)



print("\n\nResults for excitatory data: ")

path = results_path + '/data_exc'; create_folder(path) ; remove_contents(path)

data = read_and_fix_dataframe('','exc')
times,complementary_time_list = get_times(data)
exc_eeg = get_eeg(times, complementary_time_list, 'exc', '_', path)
print('excitatory spikes: ',np.sum(exc_eeg[200:]))
freqs, peaks, idx = get_frequencies(exc_eeg,'exc','_', path)



collect_data(image_selected, exc_eeg, inh_eeg, peaks,freqs,idx, seed, Lambda)

#################### results for orientations

if make_image_video == True:
    pass
else:
    quit()
    
orientations = [i*180/num_orientations for i in range(0,num_orientations)]
neuron_types = ['l_exc', 'l_inh']

for orientation_to_read in orientations:
    for exc_or_inh in neuron_types:
        
        path = results_path + '/results_' + str(orientation_to_read) + '_' + str(exc_or_inh)
        create_folder(path); remove_contents(path)

        data = read_and_fix_dataframe(orientation_to_read,exc_or_inh)

        if len(data) < 2:
            continue
        times = generate_frames(data)
        frames, complementary_time_list = generate_empty_frames(times)
        img_array = read_frames(frames)
        create_video(img_array,orientation_to_read ,exc_or_inh, path)
        create_avg_img(img_array,orientation_to_read ,exc_or_inh , path)
        #eeg = get_eeg(times, complementary_time_list, orientation_to_read, exc_or_inh, path)
        #freqs = get_frequencies(eeg,orientation_to_read,exc_or_inh, path)
        

print("\n")





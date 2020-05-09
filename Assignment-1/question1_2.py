#The assignment was done on Google Colab 

#Mounting with google Drive 
from google.colab import drive
drive.mount('/content/drive')

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:05:43 2020

@author: Priyanshi Jain
"""

import numpy as np
from scipy import ndimage
import cv2

import os
images_list = os.listdir("/content/drive/My Drive/HW-1/images")


def log_calc(image, sigma):
   return ndimage.gaussian_laplace(image, sigma)

def change_in_sigma(image):
    log_results = []
    for i in range(10):
        sigma = 1
        factor = np.power(k,i)
        sigma = sigma*factor
        convolve_image = log_calc(image,sigma)
        log_results.append(convolve_image)  
    return np.array(log_results)
        

def npa(log_results, InputImage):
    multi_npa = ndimage.rank_filter(log_results, rank = -1, size = 3)
    depth = multi_npa.shape[0]
    result = np.zeros(InputImage.shape)
    for i in range(multi_npa.shape[1]):
        for j in range(multi_npa.shape[2]):
            maxi = -30
            for d in range(depth):
                if(log_results[d][i][j]> maxi):
                    maxi = log_results[d][i][j]
            result[i][j] = maxi;
    return result 

def thresholding(npa_results, InputImage):
    interest_points = []

    flatten_array = np.ndarray.flatten(npa_results)
    flatten_array = flatten_array.reshape(-1,1)
    sorted_indices = np.argsort(flatten_array, axis = 0)

    for i in range(flatten_array.shape[0]-1,-1,-1):
        interest_cur = []
        if(len(interest_points)>=1000):
          break
        if(flatten_array[sorted_indices[i]] > 0.03):
          y = int(sorted_indices[i]%(256))
          x = int(sorted_indices[i]/256)
          interest_cur.append(x)
          interest_cur.append(y)
          interest_cur.append(InputImage[x][y])
          interest_points.append(interest_cur)
    return interest_points
    
count_files = 0
for i in images_list:                
  InputImage = cv2.imread("/content/drive/My Drive/HW-1/images/" + str(i),0)
  InputImage = InputImage/255.0
  InputImage = cv2.resize(InputImage,(256,256))
  k = 1.5            
  log_results = change_in_sigma(InputImage)
  
  npa_results = npa(log_results, InputImage)
 
  interest_points = thresholding(npa_results, InputImage)
  #print(len(interest_points))
  file_name = str(i[:len(i)-4]) + "_blob.txt"
  np.savetxt("/content/drive/My Drive/HW-1/Blob_points_final/" + file_name ,interest_points)
  count_files +=1
  if(count_files%100==0):
    print(count_files)
print(count_files)
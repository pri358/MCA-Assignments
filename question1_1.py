#The assignment was done on Google Colab 

#Moutning with google Drive 
from google.colab import drive
drive.mount('/content/drive')

#Feature Extraction

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:37:36 2020

@author: Priyanshi Jain
"""

import cv2
import numpy as np
import pickle
import os

images_list = os.listdir("/content/drive/My Drive/HW-1/images")

def thresholding(image):
  W,H,d = image.shape
  for x in range(W):
      for y in range(H):
        image[x][y][0] = int((image[x][y][0]/256.0)*6)
        image[x][y][1] = int((image[x][y][1]/256.0)*6)
        image[x][y][2] = int((image[x][y][2]/256.0)*6)
  return image



def color_array(size):
  colors = []
  for i in range(0,size):
    for j in range(0,size):
      for k in range(0,size):
        colors.append(np.array([i,j,k]))
  return colors



def preprocess(image):
  image = cv2.resize(image,(360,360))
  image_final = thresholding(image)
  dist = [i for i in range(1, 9, 2)]
  #print(np.array(result).shape)
  colors64 = color_array(6)
  
  #colors64 = unique(np.array(result))
  return image_final, dist, colors64

def correlogram(image, dist, colours):
  #print("Colors.shape " + str(colours.shape))
  W,H,d = image.shape
  correlogram = np.zeros((len(dist),len(colours)))
  for index in range(len(dist)):
    d = dist[index]
    count = 0
    gap = int(d/2)
    if(gap == 0):
      gap = 2
    for x in range(0,W, int(W/6)):
      for y in range(0,H, int(H/6)):
        current_pixel = image[x][y]
        
        for di in range(-d,d+1,gap):
          x_n = x + di
          y_n = y + d
          if(x_n>=0 and x_n < W and y_n>=0 and y_n<H):
            color_n = image[x_n][y_n]
            for c in range(len(colours)):
              if(np.array_equal(color_n,colours[c]) and np.array_equal(current_pixel,colours[c])):
                correlogram[index][c] +=1
                count += 1
                
        for di in range(-d,d+1,gap):
          x_n = x + di
          y_n = y - d
          if(x_n>=0 and x_n < W and y_n>=0 and y_n<H):
            color_n = image[x_n][y_n]
            for c in range(len(colours)):
              if(np.array_equal(color_n,colours[c]) and np.array_equal(current_pixel,colours[c])):
                correlogram[index][c] +=1
                count += 1
                
        for di in range(-d,d+1,gap):
          x_n = x + d
          y_n = y + di
          if(x_n>=0 and x_n < W and y_n>=0 and y_n<H):
            color_n = image[x_n][y_n]
            for c in range(len(colours)):
              if(np.array_equal(color_n,colours[c]) and np.array_equal(current_pixel,colours[c])):
                correlogram[index][c] +=1
                count += 1
                
        for di in range(-d,d+1,gap):
          x_n = x - d
          y_n = y + di
          if(x_n>=0 and x_n < W and y_n>=0 and y_n<H):
            color_n = image[x_n][y_n]
            for c in range(len(colours)):
              if(np.array_equal(color_n,colours[c]) and np.array_equal(current_pixel,colours[c])):
                correlogram[index][c] +=1
                count += 1
    if(count==0):
      count = 1
    for i in range(len(colours)):
      correlogram[index][i] = float(correlogram[index][i])/count
  return correlogram 



count_files = 4583
saved_list = os.listdir("/content/drive/My Drive/HW-1/Correlogram1")
for i in images_list:     
  file_name = str(i[:len(i)-4]) + "_cor.pickle"
  if file_name in saved_list:
    continue           
  InputImage = cv2.imread("/content/drive/My Drive/HW-1/images/" + str(i))
  image,dist,colours = preprocess(InputImage)
  corr = correlogram(image,dist,colours)
  file = "/content/drive/My Drive/HW-1/Correlogram1/" + file_name
  pickle_out = open(file,"wb")
  pickle.dump(corr, pickle_out)
  pickle_out.close()
  count_files +=1
  if(count_files%50==0):
    print(count_files)
print(count_files)

#Similarity Matching 

import numpy as np
import cv2 
import operator
import pickle
import time

import os
correlogram_list = os.listdir("/content/drive/My Drive/HW-1/Correlogram1")

corr_features = {}
count = 0
for i in correlogram_list:

  pickle_in = open("/content/drive/My Drive/HW-1/Correlogram1/" + i,"rb")
  corr_feature = pickle.load(pickle_in)
  count +=1
  image_name = str(i[:len(i)-11])
  # blob_point= np.loadtxt("/content/drive/My Drive/HW-1/Blob_points/" + file_name)
  corr_features.update({image_name : corr_feature})
  if(count%100==0):
    print(count)
    print(image_name)
print("Import done")


def distance(query,sample):
  diff = np.subtract(query,sample)
  ans = 0
  for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
      ans += diff[i][j]*diff[i][j]
  return ans

dist = {}
def evaluate(query_features):
  image_names = list(corr_features.keys())
  for i in range(len(image_names)):
    image_name = image_names[i]

    d = distance(query_features,corr_features.get(image_name))
    dist.update({image_name:d})
  sorted_dist = sorted(dist.items(), key=operator.itemgetter(1))

  return sorted_dist

def get_top(sorted_dist, num):
  count = 0
  top_10 = []
  for i in range(len(sorted_dist)):
    # print(sorted_dist[i])
    count += 1
    name = sorted_dist[i][0]
    top_10.append(name)
    if(count==num):
      break
  return top_10


def query_features(query_image):
  i = query_image + "_cor.pickle"
  pickle_in = open("/content/drive/My Drive/HW-1/Correlogram1/" + i,"rb")
  corr_features = pickle.load(pickle_in)
  return corr_features


query_files = os.listdir("/content/drive/My Drive/HW-1/train/query")
precision = []
recall = []
f1 = []

retrieve_good = []
retrieve_ok = []
retrieve_junk = []
time_elapsed = []
for file in query_files:
  t_start = time.clock()
  image_file = open("/content/drive/My Drive/HW-1/train/query/"+file ,'r')
  file_name = image_file.read().split()[0]
  file_name = file_name[5:]
  ground_truth = file[:len(file)-9]

  sorted_dict = evaluate(query_features(file_name))
  top = 100

  top_50_names = get_top(sorted_dict,top)

  t_end = time.clock() - t_start

  time_elapsed.append(t_end)
  ground_truth_good = open("/content/drive/My Drive/HW-1/train/ground_truth/" + ground_truth + "good.txt" ,'r')
  good_names = ground_truth_good.readlines()
  top_good = []
  for j in range(len(good_names)):
    feature = good_names[j][:len(good_names[j])-1]
    top_good.append(feature)
  actual_good = len(top_good)
  count_good = 0
  for i in top_50_names:
    if i in top_good:
      count_good +=1

  retrieve_good.append(count_good/float(actual_good))

  ground_truth_ok = open("/content/drive/My Drive/HW-1/train/ground_truth/" + ground_truth + "ok.txt" ,'r')
  ok_names = ground_truth_ok.readlines()
  top_ok = []
  for j in range(len(ok_names)):
    feature = ok_names[j][:len(ok_names[j])-1]
    top_ok.append(feature)
  actual_ok = len(top_ok)
  count_ok = 0
  for i in top_50_names:
    if i in top_ok:
      count_ok +=1

  retrieve_ok.append(count_ok/float(actual_ok))

  ground_truth_junk = open("/content/drive/My Drive/HW-1/train/ground_truth/" + ground_truth + "junk.txt" ,'r')
  junk_names = ground_truth_junk.readlines()
  top_junk = []
  for j in range(len(junk_names)):
    feature = junk_names[j][:len(junk_names[j])-1]
    top_junk.append(feature)
  actual_junk = len(top_junk)
  count_junk = 0
  for i in top_50_names:
    if i in top_junk:
      count_junk +=1

  retrieve_junk.append(count_junk/float(actual_junk))

  cur_precision = (count_good+count_ok+count_junk)/float(top)
  precision.append(cur_precision)
  cur_recall = (count_good+count_ok+count_junk)/float(actual_good+actual_junk+actual_ok)
  recall.append(cur_recall)
  cur_f1 = 2*((cur_precision*cur_recall)/float(cur_precision+cur_recall))
  f1.append(cur_f1)



#results

print("Max recall ",max(recall))
print("Min recall ",min(recall))
print("Avg recall ", sum(recall)/float(len(recall)))

print("Max precision ",max(precision))
print("Min precision ",min(precision))
print("Avg precision ", sum(precision)/float(len(precision)))

print("Max f1 ",max(f1))
print("Min f1 ",min(f1))
print("Avg f1 ", sum(f1)/float(len(f1)))

print("Avg time ",sum(time_elapsed)/float(len(time_elapsed)) )


print("Avg good ", sum(retrieve_good)/float(len(retrieve_good)))
print("Avg ok ", sum(retrieve_ok)/float(len(retrieve_ok)))
print("Avg junk ", sum(retrieve_junk)/float(len(retrieve_junk)))


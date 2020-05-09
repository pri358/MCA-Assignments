
#Spectrogram

import os
import pickle
import librosa
import numpy as np

training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/training")
spec_features = {}

for class_name in training_classes:
  print(class_name)
  audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/training/" + class_name)
  X = []
  for audio in audio_files:
    spec_file = audio + "_spec.pickle"
    pickle_in = open("/content/drive/My Drive/6th semester/Dataset/spectrogram_new/" + class_name +"/" + spec_file ,"rb")
    spectrogram = pickle.load(pickle_in)
    # print(spectrogram.shape)
    X.append(spectrogram)
  spec_features.update({class_name:X})

X_train = []
Y_train = []

for class_name in training_classes:
  print(class_name)
  X_cur = spec_features.get(class_name)
  reqd_size = 161*99
  for i in range(len(X_cur)):
    feature = np.array(X_cur[i])
    # print(feature.shape)
    feature = feature.flatten()
    # feature = feature.reshape((1,-1))
    feature = list(feature)
    # print(len(feature))
    if(len(feature) < reqd_size):
      size = (reqd_size - len(feature))
      for i in range(size):
        feature.append(0)
      # print(len(feature))
    elif(len(feature) > reqd_size):
      feature = feature[:reqd_size]
    feature = list(feature)
    X_train.append(feature)
    if(len(feature) !=reqd_size):
      print(len(feature))
    if(class_name == "zero"):
      Y_train.append(0)
    elif(class_name == "one"):
      Y_train.append(1)
    elif(class_name == "two"):
      Y_train.append(2)
    elif(class_name == "three"):
      Y_train.append(3)
    elif(class_name == "four"):
      Y_train.append(4)
    elif(class_name == "five"):
      Y_train.append(5)
    elif(class_name == "six"):
      Y_train.append(6)
    elif(class_name == "seven"):
      Y_train.append(7)
    elif(class_name == "eight"):
      Y_train.append(8)
    elif(class_name == "nine"):
      Y_train.append(9)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape)
print(Y_train.shape)

import os
import pickle
import librosa
import numpy as np

training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/training")

X_val = []
Y_val = []

for class_name in training_classes:
  print(class_name)
  class_no = 0
  if(class_name == "zero"):
    class_no = 0
  elif(class_name == "one"):
    class_no = 1
  elif(class_name == "two"):
    class_no = 2
  elif(class_name == "three"):
    class_no = 3
  elif(class_name == "four"):
    class_no = 4
  elif(class_name == "five"):
    class_no = 5
  elif(class_name == "six"):
    class_no = 6
  elif(class_name == "seven"):
    class_no = 7
  elif(class_name == "eight"):
    class_no = 8
  elif(class_name == "nine"):
    class_no = 9
  audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/validation/" + class_name)
  for audio in audio_files:
    spec_file = audio + "_spec.pickle"
    pickle_in = open("/content/drive/My Drive/6th semester/Dataset/val_spec/" + class_name +"/" + spec_file ,"rb")
    spectrogram = pickle.load(pickle_in)
    reqd_size = 161 * 99 
    spectrogram = list(spectrogram.flatten())
    if(len(spectrogram) < reqd_size):
      size = reqd_size - len(spectrogram)
      for i in range(size):
        spectrogram.append(0)
    X_val.append(spectrogram)
    Y_val.append(class_no)



X_val = np.array(X_val)
Y_val = np.array(Y_val)
print(X_val.shape)
print(Y_val.shape)  


from sklearn import svm
clf = svm.SVC(kernel = 'rbf')
# X_train = np.array(X_train)
# print(X_train.shape)
clf.fit(X_train,Y_train)
from sklearn.metrics import classification_report
print(classification_report(Y_val, clf.predict(X_val)))
print("train " , clf.score(X_train, Y_train))
print("val", clf.score(X_val, Y_val))
  


#MFCC
import os
import pickle
import librosa
import numpy as np

training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/training")

X_train = []
Y_train = []

for class_name in training_classes:
  print(class_name)
  class_no = 0
  if(class_name == "zero"):
    class_no = 0
  elif(class_name == "one"):
    class_no = 1
  elif(class_name == "two"):
    class_no = 2
  elif(class_name == "three"):
    class_no = 3
  elif(class_name == "four"):
    class_no = 4
  elif(class_name == "five"):
    class_no = 5
  elif(class_name == "six"):
    class_no = 6
  elif(class_name == "seven"):
    class_no = 7
  elif(class_name == "eight"):
    class_no = 8
  elif(class_name == "nine"):
    class_no = 9
  audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/training/" + class_name)
  for audio in audio_files:
    mfcc_file = audio + "_mfcc.pickle"
    pickle_in = open("/content/drive/My Drive/6th semester/Dataset/mfcc/" + class_name +"/" + mfcc_file ,"rb")
    mfcc = pickle.load(pickle_in)
    # print(mfcc.shape)
    reqd_size = 99 * 12 
    mfcc = list(mfcc.flatten())
    if(len(mfcc) < reqd_size):
      size = reqd_size - len(mfcc)
      for i in range(size):
        mfcc.append(0)
    if(len(mfcc) != reqd_size):
      print(len(mfcc))
    X_train.append(mfcc)
    Y_train.append(class_no)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape)
print(Y_train.shape)

print("Train imported ")



import os
import pickle
import librosa
import numpy as np

training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/training")

X_val = []
Y_val = []

for class_name in training_classes:
  print(class_name)
  class_no = 0
  if(class_name == "zero"):
    class_no = 0
  elif(class_name == "one"):
    class_no = 1
  elif(class_name == "two"):
    class_no = 2
  elif(class_name == "three"):
    class_no = 3
  elif(class_name == "four"):
    class_no = 4
  elif(class_name == "five"):
    class_no = 5
  elif(class_name == "six"):
    class_no = 6
  elif(class_name == "seven"):
    class_no = 7
  elif(class_name == "eight"):
    class_no = 8
  elif(class_name == "nine"):
    class_no = 9
  audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/validation/" + class_name)
  for audio in audio_files:
    mfcc_file = audio + "_mfcc.pickle"
    pickle_in = open("/content/drive/My Drive/6th semester/Dataset/val_mfcc/" + class_name +"/" + mfcc_file ,"rb")
    mfcc = pickle.load(pickle_in)
    reqd_size = 99 * 12
    mfcc = list(mfcc.flatten())
    if(len(mfcc) < reqd_size):
      size = reqd_size - len(mfcc)
      for i in range(size):
        mfcc.append(0)
    if(len(mfcc) != reqd_size):
      print(len(mfcc))
    X_val.append(mfcc)
    Y_val.append(class_no)



X_val = np.array(X_val)
Y_val = np.array(Y_val)
print(X_val.shape)
print(Y_val.shape)  

print("Validation imported ")

from sklearn import svm
clf = svm.SVC(kernel = 'rbf')
# X_train = np.array(X_train)
# print(X_train.shape)
clf.fit(X_train,Y_train)

from sklearn.metrics import classification_report
print(classification_report(Y_val, clf.predict(X_val)))
print("train ", clf.score(X_train, Y_train))
print("val", clf.score(X_val, Y_val))



#data augmentation 

def data_aug(sample,noise):
  if(len(noise)>len(sample)):
    noise = noise[:len(sample)]
  elif(len(noise) < len(sample)):
    for i in range(len(sample) - len(noise)):
      noise.append(0)
  mix_sample = sample+noise
  return mix_sample

 #Spectrogram 

import sklearn
import numpy as np

import cmath
import numpy as np
from math import log, ceil


def spectrogram(samples, sample_rate, window_ms = 20.0):

    overlap_size = 0.5
    window_size = int(0.001 * sample_rate * window_ms)
    stride_size = int(window_size*overlap_size)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size

    samples = samples[:len(samples) - truncate_size]

    cols = int((len(samples) - window_size) / stride_size) + 1

    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape = (window_size,cols), strides = nstrides)
    

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    windows = windows*weighting 
    fft1 = np.fft.rfft(windows, axis = 0)
    fft1 = np.absolute(fft1)
    fft1 = fft1**2
    fft1 = sklearn.preprocessing.normalize(fft1)
    return fft1

 import os
import pickle
import numpy as np 
import librosa

training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/validation")
noise_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/_background_noise_")
noise_file = noise_files[np.random.randint(0,len(noise_files))]
noise,n_rate = librosa.load("/content/drive/My Drive/6th semester/Dataset/_background_noise_/" + noise_file, sr = None, mono = True, offset = 0.0, duration = None)

for class_name in training_classes:
  audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/training/" + class_name)
  print(class_name)
  noise_number = 0.4*len(audio_files)
  count_noise = 0
  count = 0
  
  for audio in audio_files:
    file_path = "/content/drive/My Drive/6th semester/Dataset/training/" + class_name + "/" + audio
    samples, sampling_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)
    if(count_noise<noise_number):
      samples = data_aug(samples,noise)
      count_noise += 1
    spect = spectrogram(samples, sampling_rate)

    file_name = audio + "_spec.pickle"
    file = "/content/drive/My Drive/6th semester/Dataset/spec_aug/" + class_name + "/" + file_name
    pickle_out = open(file,"wb")
    pickle.dump(spect, pickle_out)
    pickle_out.close()
    count += 1
  print(count)

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
    pickle_in = open("/content/drive/My Drive/6th semester/Dataset/spec_aug/" + class_name +"/" + spec_file ,"rb")
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

from scipy.fftpack import dct
import numpy as np
def emphasis_phase(samples, sample_rate):
  emphasis_coef = 0.95
  emph_signal = []
  emph_signal.append(samples[0])
  for i in range(1,samples.shape[0]):
    emph_signal.append(samples[i] - emphasis_coef*samples[i-1])
  emph_signal = np.array(emph_signal)
  return emph_signal
  # print(emph_signal.shape)

def spectrogram(samples, sample_rate, window_ms = 20.0, n = 512):

    overlap_size = 0.5
    window_size = int(0.001 * sample_rate * window_ms)
    stride_size = int(window_size*overlap_size)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size

    samples = samples[:len(samples) - truncate_size]
    
    cols = int((len(samples) - window_size) / stride_size) + 1

    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape = (window_size,cols), strides = nstrides)
    

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, n, axis=0)
    fft = np.absolute(fft)/n
    fft = np.power(fft,2)
    return fft


def meltohz(mel_points):
    return 700 * (10**(mel_points / 2595) - 1)

def get_filter_banks(spect, sample_rate, n = 512):
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    n_filt  = 40
    melpoints = np.linspace(low_mel,high_mel, n_filt+2)
    hzpoints = meltohz(melpoints)
    bin = np.floor((n+1)*hzpoints/sample_rate)

    fbank = np.zeros([n_filt,int(n/2)+1])
    # print(fbank.shape)
    for j in range(0,n_filt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    
    filter_banks = np.dot(spect.T, fbank.T)
    # print(filter_banks.shape)
    for i in range(filter_banks.shape[0]):
      for j in range(filter_banks.shape[1]):
        if(filter_banks[i][j] == 0):
          filter_banks[i][j] = np.finfo(np.float32).eps
    filter_banks = 10 * np.log10(filter_banks)

    return filter_banks


def final_mfcc(filter_banks):
    
  numcep = 12
  mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
  mfcc = mfcc[:,:numcep]

  n = np.arange(mfcc.shape[1])
  
  x = np.sin(np.pi * n / 22)
  lift = 1 + (22/ 2) * x
  mfcc *= lift

  mfcc = sklearn.preprocessing.normalize(mfcc)
  return mfcc

import os
import pickle
import librosa
training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/training")
noise_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/_background_noise_")
noise_file = noise_files[np.random.randint(0,len(noise_files))]
noise,n_rate = librosa.load("/content/drive/My Drive/6th semester/Dataset/_background_noise_/" + noise_file, sr = None, mono = True, offset = 0.0, duration = None)

for class_name in training_classes:
  if(class_name == "nine" or class_name == "eight"):
    print(class_name)
    count = 0
    audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/training/" + class_name)
    noise_number = 0.4*len(audio_files)
    count_noise = 0
    for audio in audio_files:
      
      file_path = "/content/drive/My Drive/6th semester/Dataset/training/"+ class_name+ "/" + audio
      sample, sample_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)
      if(count_noise<noise_number):
        sample = data_aug(sample,noise)
        count_noise += 1 
      emph_signal = emphasis_phase(sample,sample_rate)
      spect = spectrogram(emph_signal, sample_rate)
      filter_banks = get_filter_banks(spect, sample_rate)
      mfcc = final_mfcc(filter_banks)

      file_name = audio + "_mfcc.pickle"
      file = "/content/drive/My Drive/6th semester/Dataset/mfcc_aug/" + class_name + "/" + file_name
      pickle_out = open(file,"wb")
      pickle.dump(mfcc, pickle_out)
      pickle_out.close()
      count += 1
    print(count)

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
    pickle_in = open("/content/drive/My Drive/6th semester/Dataset/mfcc_aug/" + class_name +"/" + mfcc_file ,"rb")
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

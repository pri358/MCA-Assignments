import sklearn
import numpy as np



def spectrogram(samples, sample_rate, window_ms = 20.0):

    overlap_size = 0.5
    window_size = int(0.001 * sample_rate * window_ms)
    stride_size = int(window_size*overlap_size)

    truncate_size = (len(samples) - window_size) % stride_size

    samples = samples[:len(samples) - truncate_size]

    cols = int((len(samples) - window_size) / stride_size) + 1
    
    windows = np.lib.stride_tricks.as_strided(samples, shape = (window_size,cols), strides = (samples.strides[0], samples.strides[0] * stride_size))
    

    weighting = np.hanning(window_size)
    weighting = weighting[:, None]
    windows = windows*weighting 
    fft1 = np.absolute(np.fft.rfft(windows,axis = 0))
    print(fft1.shape)
    fft1 = fft1**2
    fft1 = sklearn.preprocessing.normalize(fft1)
    return fft1

 #saving training features to drive

import os
import pickle
import librosa 

training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/training")

for class_name in training_classes:
  print(class_name)
  count = 0
  audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/training/" + class_name)
  for audio in audio_files:
    file_path = "/content/drive/My Drive/6th semester/Dataset/training/" + class_name + "/" + audio
    samples, sampling_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)
    spect = spectrogram(samples, sampling_rate)

    file_name = audio + "_spec.pickle"
    file = "/content/drive/My Drive/6th semester/Dataset/spectrogram_new/" + class_name + "/" + file_name
    pickle_out = open(file,"wb")
    pickle.dump(spect, pickle_out)
    pickle_out.close()
    count += 1
  print(count)


#saving validation features to drive

training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/validation")

for class_name in training_classes:
  print(class_name)
  count = 0
  audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/validation/" + class_name)
  for audio in audio_files:
    file_path = "/content/drive/My Drive/6th semester/Dataset/validation/" + class_name + "/" + audio
    samples, sampling_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)
    spect = spectrogram(samples, sampling_rate)

    file_name = audio + "_spec.pickle"
    file = "/content/drive/My Drive/6th semester/Dataset/val_spec/" + class_name + "/" + file_name
    pickle_out = open(file,"wb")
    pickle.dump(spect, pickle_out)
    pickle_out.close()
    count += 1
  print(count)

  #discussed with Pranjal Kaura
  
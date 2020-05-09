from scipy.fftpack import dct
import numpy as np
import sklearn

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
    fft = fft ** 2
    return fft


def meltohz(mel_points):
    return 700 * (10**(mel_points / 2595) - 1)

def get_filter_banks(spect, sample_rate, n = 512):
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    n_filt  = 40
    melpoints = np.linspace(low_mel,high_mel, n_filt+2)
    hzpoints = meltohz(melpoints)
    bin = int((n+1)*hzpoints/sample_rate)

    fbank = np.zeros([n_filt,int(n/2)+1])

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

#saving features 

import os
import pickle
import librosa
training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/training")

# for class_name in training_classes:
print("zero")
count = 0
audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/training/zero")
for audio in audio_files:
  file_path = "/content/drive/My Drive/6th semester/Dataset/training/zero"+ "/" + audio
  sample, sample_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)

  emph_signal = emphasis_phase(sample,sample_rate)
  spect = spectrogram(emph_signal, sample_rate)
  filter_banks = get_filter_banks(spect, sample_rate)
  mfcc = final_mfcc(filter_banks)

  file_name = audio + "_mfcc.pickle"
  file = "/content/drive/My Drive/6th semester/Dataset/mfcc/" + "zero" + "/" + file_name
  pickle_out = open(file,"wb")
  pickle.dump(mfcc, pickle_out)
  pickle_out.close()
  count += 1
print(count)

#saving validation features to drive

import os
import pickle
import librosa
training_classes = os.listdir("/content/drive/My Drive/6th semester/Dataset/validation")

for class_name in training_classes:
  print(class_name)
  count = 0
  audio_files = os.listdir("/content/drive/My Drive/6th semester/Dataset/validation/" + class_name)
  for audio in audio_files:
    file_path = "/content/drive/My Drive/6th semester/Dataset/validation/" + class_name + "/" + audio
    sample, sample_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)

    emph_signal = emphasis_phase(sample,sample_rate)
    spect = spectrogram(emph_signal, sample_rate)
    filter_banks = get_filter_banks(spect, sample_rate)
    mfcc = final_mfcc(filter_banks)

    file_name = audio + "_mfcc.pickle"
    file = "/content/drive/My Drive/6th semester/Dataset/val_mfcc/" + class_name + "/" + file_name
    pickle_out = open(file,"wb")
    pickle.dump(mfcc, pickle_out)
    pickle_out.close()
    count += 1
  print(count)

  #discussed with Pranjal Kaura 
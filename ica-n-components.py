import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.signal as sps
from IPython.display import Audio
import os
import sys
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import FastICA
from mcc import mean_corr_coef as mcc

mypath = os.getcwd() + '/' + sys.argv[1]
audionames = []
for file in os.listdir(mypath):
    if file.endswith(".wav"):
        audionames.append(file)

# Load audios from folder. Make sure that all audios have same sampling rate
sampling_rate = 44100 # Define sampling rate here
print('Loading audios...')
audios = []
for audio in audionames:
    f, x = scipy.io.wavfile.read(mypath + '/' + audio)
    if f != sampling_rate:
        n_samples = round(len(x) * float(sampling_rate) / f)
        x = sps.resample(x, n_samples)
    audios.append(x)
print('Ready.\n')

preprocessed = []
audio_lengths = []
# Pre-process audios: Pick only one channel
print('Pre-processing audios...')
for audio in audios:
    # Pick only first channel for simplicity
    if len(audio.shape) > 1:
        audio = audio[:,0]

    preprocessed.append(audio)
    audio_lengths.append(len(audio))

# Cut audios to have same length 
min_length = min(audio_lengths)
cropped_audios = []
for audio in preprocessed:
    audio = audio[:min_length]
    cropped_audios.append(audio)
print('Ready.\n')

# Visualize original sources
print('Plotting original source waveforms...')
i = 0
for audio in cropped_audios:
    # Plot waveforms
    #plt.figure()
    plt.plot(audio)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Waveform of source '+ str(i+1))
    #plt.savefig(mypath + '/' + 'orig_waveform' + str(i+1) + '.png')
    i += 1
    #plt.show()

# Do ICA
print('Doing ICA...')
X = np.array(cropped_audios)
transformer = FastICA(n_components=len(audionames), random_state=0, whiten=True)
S_est = transformer.fit_transform(X.T)
print('Ready.\n')

# Make folder for estimated sources 
if not os.path.exists(mypath + '/estimated_sources'):
    os.makedirs(mypath + '/estimated_sources')

# Visualize and save estimated sources
print('Plotting estimated source waveforms...\n')
for i in range(S_est.shape[1]):
    s = S_est[:,i]

    # plot waveform
    #plt.figure()
    plt.plot(s)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Estimated source %d ' % (i+1))
    #plt.savefig(mypath + '/estimated_src_waveform' + str(i+1) + '.png')
    #plt.show()

    # Save estimated source
    #scipy.io.wavfile.write(mypath + '/estimated_sources/estimated_src' + str(i+1) + '.wav', sampling_rate, s.astype(np.int16))
    scipy.io.wavfile.write(mypath + '/estimated_sources/estimated_src' + str(i+1) + '.wav', sampling_rate, s)

print('All finished! :-) You can find your separated audios in ', mypath + '/estimated_sources.')

#Load S
# Evaluate performance: get mean correlation coefficient
#print('Mean correlation coefficient: ',  mcc(S_est, S))


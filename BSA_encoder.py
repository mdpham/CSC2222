import argparse
import os
from spikes import encoder
from pprint import pprint
import numpy as np
import mne
import pathlib

from spikes.utility import ReadCSV

parser = argparse.ArgumentParser(description=".cnt EEG preprocessing: BSA encoding")
parser.add_argument("-v", dest="verbose", help="verbose", action='store_true')
parser.add_argument("-file", dest="filepath", help="input cnt file", type=pathlib.Path, default='./EEGparticipant 015.cnt')
parser.add_argument('-ec', dest='exclude_channels', help='channel names to exclude', nargs="+", type=str, default=[])
parser.add_argument('-bsa_t', dest='threshold', help='BSA threshold parameter', type=float, default=0.6)
parser.add_argument('-bsa_fl', dest='filter_length', help='scipy.signal.firwin filter_length', type=int, default=2)
parser.add_argument('-bsa_c', dest='cutoff', help='scipy.signal.firwin cutoff', type=float, default=0.8)
parser.add_argument('-l_freq', dest='filter_low_freq', help='mne.filter frequency lower bound', type=float, default=0.1)
parser.add_argument('-h_freq', dest='filter_high_freq', help='mne.filter frequency upper bound', type=float, default=30)

args = parser.parse_args()

verbose = args.verbose
# .cnt arguments
filepath = args.filepath
exclude_channels = args.exclude_channels
# Frequency filtering arguments
l_freq = args.filter_low_freq
h_freq = args.filter_high_freq
# BSA encoding arguments
threshold = args.threshold
filter_length = args.filter_length
cutoff = args.cutoff

mne.set_log_level(verbose)



sample_dir = pathlib.Path(str(filepath.with_suffix(''))+'_epochs')
if os.path.exists(sample_dir):
    if verbose: print('Pyspikes data directory at '+str(sample_dir)+' already exists, skipping writing sam*.csv files')
    # shutil.rmtree(sample_dir)
else:
    if verbose: print('Creating new pyspikes data directory for encoding')
    if verbose: print('Loading .cnt file from '+str(filepath))

    raw_data = mne.io.read_raw_cnt(filepath, preload=True)
    sampling_freq = raw_data.info['sfreq']
    
    if verbose:
        print('Channel names:')
        print(*('\t'+ch_name for ch_name in raw_data.info['ch_names']), sep=' ')
        print('Channels to exclude:')
        print(*('\t'+ch for ch in exclude_channels), sep=' ')
        print('Sampling frequency: %s' % sampling_freq)
    
    # Use build in MNE methods
    raw_data.filter(l_freq=l_freq, h_freq=h_freq)
    epochs = mne.make_fixed_length_epochs(raw_data, duration=1, preload=True)
    epochs.drop_channels(exclude_channels)
    os.mkdir(sample_dir)
    for i in range(len(epochs)):
        sample_path = sample_dir.joinpath('sam'+str(i)+'.csv')
        # Note: pyspikes require timepoints-by-features so transpose
        np.savetxt(sample_path, epochs[i].get_data()[0, :, :].T, delimiter=",")
    if verbose:
        print('%s epochs, %s channels, %s time points' % np.shape(epochs))



sample_data = ReadCSV(path=sample_dir).get_samples()['samples']
if verbose:
    print(np.shape(sample_data))
    print('%s BSA threshold, %s filter length,  %s cutoff' % (threshold, filter_length, cutoff))
data_encoder = encoder.BSA(sample_data, threshold=threshold, filter_length=filter_length, cutoff=cutoff)
data_spikes = data_encoder.get_spikes()

np.save(filepath.with_suffix('.npy'), data_spikes)

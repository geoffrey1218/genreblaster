import matplotlib.pyplot as plt
import numpy as np
import sunau
from scipy import signal
from scipy.io import wavfile

def stereo_to_mono(audiodata):
    '''
    From https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
    '''
    newaudiodata = []

    for i in range(len(audiodata)):
        d = (audiodata[i][0] + audiodata[i][1])/2
        newaudiodata.append(d)

    return np.array(newaudiodata, dtype='int16')

def get_spectogram(filename):
    # sample_rate, samples = wavfile.read(filename)
    # samples = stereo_to_mono(samples)
    # wavfile.write('test_mono.wav', sample_rate, samples)
    f = sunau.open(filename, 'r')
    num_frames = f.getnframes()
    sample_rate = f.getframerate()
    audio_data = np.fromstring(f.readframes(num_frames), dtype=np.int16) 
    frequencies, times, spectogram = signal.spectrogram(audio_data, sample_rate)

    plt.pcolormesh(times, frequencies, spectogram)
    plt.imshow(spectogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()



if __name__ == '__main__':
    get_spectogram('test.au')
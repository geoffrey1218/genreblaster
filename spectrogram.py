import matplotlib.pyplot as plt
import numpy as np
import sunau
from scipy import signal
from scipy.io import wavfile

def get_spectrogram(filename):
    '''
    Help from https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
    '''
    f = sunau.open(filename, 'r')
    num_frames = f.getnframes()
    sample_rate = f.getframerate()
    audio_data = np.fromstring(f.readframes(num_frames), dtype=np.int16) 
    frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()



if __name__ == '__main__':
    get_spectrogram('test.au')
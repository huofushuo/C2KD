import multiprocessing
import os
import os.path
import pickle

import librosa
import numpy as np
from scipy import signal


def audio_extract(path, audio_name, audio_path, sr=16000):
    save_path = path
    samples, samplerate = librosa.load(audio_path)
    resamples = np.tile(samples, 10)[:sr*10]
    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)
    assert spectrogram.shape == (257, 1004)
    save_name = os.path.join(save_path, audio_name + '.pkl')
    with open(save_name, 'wb') as fid:
        pickle.dump(spectrogram, fid)


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('{}: Exiting'.format(proc_name))
                self.task_queue.task_done()
                break
            # print(next_task)
            audio_extract(next_task[0], next_task[1], next_task[2])
            self.task_queue.task_done()


if __name__ == '__main__':


    print("ok")

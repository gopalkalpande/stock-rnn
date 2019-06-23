import numpy as np
import utils

class Loader:

    def __init__(self, filename, window_size, LogReturn = True):

        close = np.genfromtxt(filename, delimiter = ',', skip_header = 1, usecols = (4))

        if (LogReturn):
            log_return = utils.logret(close) 
        else:
            log_return = close

        self.train_size = log_return.shape[0] // window_size

        log_return = log_return[:self.train_size * window_size]
        self.X = log_return.reshape(self.train_size, window_size)
from tensorflow.keras.callbacks import Callback
from math import pi
from math import cos
from math import floor
from tensorflow.keras import backend
class CosineAnnealingLRS(Callback):
    # constructor
    def __init__(self, n_epochs, n_cycles, lrate_max,test_data,OUTPUT_PATH, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()
        self.pred_list = list()
        self.OUTPUT_PATH = OUTPUT_PATH
        self.test_X = test_data
        # calculate learning rate for an epoch
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)
    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs=None):
        # calculate learning rate
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)
    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        # check if we can save model
        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            pred =self.model.predict(self.test_X)# saving prediction
            self.pred_list.append([pred])
            # save model to file
            filename = self.OUTPUT_PATH+"snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
            self.model.save(filename)
            print('\n>saved snapshot %s, epoch %d' % (filename, epoch))
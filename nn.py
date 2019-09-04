import random
import numpy as np
import cv2
import os
import itertools
import math
import matplotlib.pyplot as plt
import theano.tensor as T
 
#from setup_GPU import setup_theano
#setup_theano()
 
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import sigmoid
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
 

 
def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
 

def write_image(img, path):
    return cv2.imwrite(path, img)

 
def image_matrix(img):
    if img.shape[0] == 258:
        return (img[0:258, 0:540] / 255.0).astype('float32').reshape((1, 1, 258, 540))
    if img.shape[0] == 420:
        result = []
        result.append((img[0:258, 0:540] / 255.0).astype('float32').reshape((1, 1, 258, 540)))
        result.append((img[162:420, 0:540] / 255.0).astype('float32').reshape((1, 1, 258, 540)))
        result = np.vstack(result).astype('float32').reshape((2, 1, 258, 540))
    return result


def load_train_set(file_list):
    xs = []
    ys = []
    for fname in file_list:
        x = image_matrix(load_image(os.path.join('/home/sandeep/Desktop/Major/train', fname)))
        y = image_matrix(load_image(os.path.join('/home/sandeep/Desktop/Major/train_cleaned', fname)))
        for i in range(0, x.shape[0]):
            xs.append(x[i, :, :, :].reshape((1, 1, 258, 540)))
            ys.append(y[i, :, :, :].reshape((1, 1, 258, 540)))
    return np.vstack(xs), np.vstack(ys)
 

def load_test_file(fname, folder):
    xs = []
    x = image_matrix(load_image(os.path.join(folder, fname)))
    for i in range(0, x.shape[0]):
        xs.append(x[i, :, :, :].reshape((1, 1, 258, 540)))
    return np.vstack(xs)

 
def list_images(folder):
    included_extentions = ['jpg','bmp','png','gif' ]
    results = [fn for fn in os.listdir(folder) if any([fn.endswith(ext) for ext in included_extentions])]
    return results

 
def do_test(inFolder, outFolder, nn):
    test_images = list_images(inFolder)
    nTest = len(test_images)
    for x in range(0, nTest):
        fname = test_images[x]
        x1 = load_test_file(fname, inFolder)
        x1 = x1 - 0.5
        pred_y = nn.predict(x1)
        tempImg = []
        if pred_y.shape[0] == 1:
        	tempImg = pred_y[0, 0, :, :].reshape(258, 540)
        if pred_y.shape[0] == 2:
        	tempImg1 = pred_y[0, 0, :, :].reshape(258, 540)
        	tempImg2 = pred_y[1, 0, :, :].reshape(258, 540)
        	tempImg = np.empty((420, 540))
        	tempImg[0:258, 0:540] = tempImg1
        	tempImg[162:420, 0:540] = tempImg2
        	tempImg[tempImg < 0] = 0
        	tempImg[tempImg > 1] = 1
        print fname
        tempImg = np.asarray(tempImg*255.0, dtype=np.uint8)
        write_image(tempImg, (os.path.join(outFolder, fname)))
 

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
	self.best_weights = None
 
    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
	    self.best_valid_epoch = current_epoch
	    self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
            self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()
    
def main():
        random.seed(1234)
 
training_images = list_images("/home/sandeep/Desktop/Major/train")    
random.shuffle(training_images)
nTraining = len(training_images)
TRAIN_IMAGES = training_images
 
train_x, train_y = load_train_set(TRAIN_IMAGES)
test_x = train_x
test_y = train_y
 
# centre on zero - has already been divided by 255
train_x = train_x - 0.5
 
net2 = NeuralNet(
layers = [
('input', layers.InputLayer),
('conv1', layers.Conv2DLayer),
('conv2', layers.Conv2DLayer),
('conv3', layers.Conv2DLayer),
('conv4', layers.Conv2DLayer),
('output', layers.FeaturePoolLayer),
],
#layer parameters:
input_shape = (None, 1, 258, 540),
conv1_num_filters = 15, conv1_filter_size = (7, 7), conv1_pad = 'same',
conv2_num_filters = 15, conv2_filter_size = (7, 7), conv2_pad = 'same',
#conv3_num_filters = 15, conv3_filter_size = (7, 7), conv3_pad = 'same',
output_pool_size = 15,
output_pool_function = T.sum,
y_tensor_type=T.tensor4,
 
#optimization parameters:
update = nesterov_momentum,
update_learning_rate = 0.05,
update_momentum = 0.9,
regression = True,
max_epochs = 50,
verbose = 1,
batch_iterator_train=BatchIterator(batch_size=25),
on_epoch_finished=[EarlyStopping(patience=20),],
train_split=TrainSplit(eval_size=0.5)
)
 
net2.fit(train_x, train_y)
 
plot_loss(net2)
plt.savefig("/home/sandeep/Desktop/Major/Results/plotloss.png")
plot_conv_weights(net2.layers_[1], figsize=(4, 4))
plt.savefig("/home/sandeep/Desktop/Major/Results/convweights.png")
 
#layer_info = PrintLayerInfo()
#layer_info(net2)
 
import cPickle as pickle
with open('/home/sandeep/Desktop/Major/Results/net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)
 
y_pred2 = net2.predict(test_x)
print "The accuracy of this network is: %0.2f" % (abs(y_pred2 - test_y)).mean()
 
do_test("/home/sandeep/Desktop/Major/train", '/home/sandeep/Desktop/Major/new_train_cleaned', net2)
do_test("/home/sandeep/Desktop/Major/test", '/home/sandeep/Desktop/Major/new_output', net2)
 
#if __name__ == '__main__':
main()

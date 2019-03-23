from skimage import io
from scipy.ndimage.morphology import binary_fill_holes as imfill
from scipy.io import loadmat
import numpy as np
import matplotlib
from PIL import Image
import random
import cv2
import os

chess_path = './dataset/crop_chessboards/'
cake_path = './dataset/crop_green_bean_cakes/'
paper_path = './dataset/crop_manuscript_papers/'

chess_files = os.listdir(chess_path)
cake_files = os.listdir(cake_path)
paper_files = os.listdir(paper_path)

# create labels and add to a list
chess_data = []
cake_data = []
paper_data = []

for file in chess_files:
    chess_arr = io.imread(chess_path+file)
    if len(chess_arr.shape) == 3:
        label = np.array([1,0,0])
        chess_data.append(np.array([chess_arr, label]))

for file in cake_files:
    cake_arr = io.imread(cake_path+file)
    if len(cake_arr.shape) == 3:
        label = np.array([0,1,0])
        cake_data.append(np.array([cake_arr, label]))

for file in paper_files:
    paper_arr = io.imread(paper_path+file)
    if len(paper_arr.shape) == 3:
        label = np.array([0,0,1])
        paper_data.append(np.array([paper_arr, label]))

# change to numpy array
chess_data = np.array(chess_data)
cake_data = np.array(cake_data)
paper_data = np.array(paper_data)

# randomly shuffle data
np.random.shuffle(chess_data)
np.random.shuffle(cake_data)
np.random.shuffle(paper_data)

training_ratio = 0.75

# get training data
training_chess = chess_data[0: int(len(chess_data)*training_ratio)]
training_cake = cake_data[0: int(len(cake_data)*training_ratio)]
training_paper = paper_data[0: int(len(paper_data)*training_ratio)]

# get tast data
test_chess = chess_data[int(len(chess_data)*training_ratio):]
test_cake = cake_data[int(len(cake_data)*training_ratio):]
test_paper = paper_data[int(len(paper_data)*training_ratio):]

# concat all training data
training_data = np.concatenate([training_chess, training_cake, training_paper],axis=0)
np.random.shuffle(training_data)
print('training data shape: ', training_data.shape)

# concat all test data
test_data = np.concatenate([test_chess, test_cake, test_paper],axis=0)
print('test data shape: ', test_data.shape)


# create data handler
class DataHandler():
    def __init__(self, training_data, test_data):
        self.tr_data = training_data
        self.ts_data = test_data
        self.pointer = 0
        self.epoch = 0
        self.data_index = np.arange(len(self.tr_data))
        
    def next_batch(self,num):
        if self.pointer + num <= len(self.tr_data):
            index = self.data_index[self.pointer:self.pointer+num]
            self.pointer += num
        else:
            new_pointer = self.pointer + num - len(self.tr_data)
            index = np.concatenate((self.data_index[self.pointer:], self.data_index[:new_pointer]),axis=0)
            self.shuffle_data()
            self.pointer = new_pointer
        
        batch_samples = np.array([ i[0] for i in self.tr_data[index]])
        batch_labels = np.array([ i[1] for i in self.tr_data[index]])
        
        return batch_samples, batch_labels
     
    def shuffle_data(self):
        np.random.shuffle(self.data_index)
        self.epoch += 1
        #print('epoch: ', self.epoch)
        
    def reset(self):
        self.pointer = 0
        self.data_index = np.arange(len(self.tr_data))
        self.epoch = 0
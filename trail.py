__author__ = 'dilip'
import numpy as np
import image
import cv2
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

example =cv2.imread("texture1.tif",cv2.IMREAD_GRAYSCALE)
##example = example1[:, :, 0]
##print (example.shape)
##print (type(example))
##print(example.ndim)
##print(example.shape)
##example=example.reshape(256*256,1)
print(example.shape)
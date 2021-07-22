# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:34:49 2021

@author: SYAVIRA TZ
"""

import csv
import numpy as np
import random
import math
import operator

# Function load dataset
#   Parameters: 
#       filename: file that can we used: image_segmentation.txt 
#           source: http://archive.ics.uci.edu/ml/datasets/image+segmentation, with modified at the class columns, change categorical to numeric
#       no_split: type float, contain the number for devide data into 2 set (training and testing)
#       trainingSet: type list of the splitted training data
#       testingSet: typle list of the splitted testing data
# Return dataset, testSet, and trainingSet
def Load(filename, no_split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = np.loadtxt(csvfile, delimiter=",", skiprows = 1) #load dataset using numpy, skip first row
        #Split dataset into training and testing randomly with no_split as threshold
        for row in lines:
            dom = random.random()
            if dom < no_split:
                trainingSet.append(row)
            else:
                testSet.append(row)
        return lines
    
# Fuction euclideanDistance: for calculate the distance of training data and training instance
#   Parameters:
#       instance1: observation value
#       instance2: neighbor, 
#       length: the total comparing
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += np.power((instance1[x] - instance2[x]), 2)
	return np.sqrt(distance)

# Function cosine similarity metric : for calculate the distance pf training data and training instance
#   Parameters:
#       instance1: observation value
#       instance2: neighbor, 
#       length: the total comparing
def cosine(instance1, instance2, length):
    sum1, sum2, sum3 = 0, 0, 0
    for i in range(length):
        sum1 += instance1[i]*instance2[i]   # A o B
        sum2 += instance1[i]*instance1[i]    
        sum3 += instance2[i]*instance2[i]   
    sum2 = math.sqrt(sum2)                  # ||A||
    sum3 = math.sqrt(sum3)                  # ||B||
    cos = float(sum1)/float(sum2 * sum3)    # find cos value
    return (1- cos)                         # distance = 1 - cos

# Function fit: to get neighbors of based on k value
#   Parameters:
#       trainingSet: type list of the splitted training data
#       testInstance: the value of each testing data
#       k: define the number k of neighbor
def fit(trainingSet, testInstance, k):
    distances = [] # Store actual class of trainingSet and distance
    trainingCol = np.size(trainingSet,1) # Store number of columns in the trainingSet
    
    length = len(testInstance)-1 #length : the total comparing of neighbor
    for x in range(len(trainingSet)):
        dist = cosine(testInstance, trainingSet[x], length) #Calculate the distance
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Function predict: to calculate mean of the neighbor's distance.
# This step is difference with classification. If classification, use majority to make prediction,
# but in regression it take the average of the target variable of the k nearest neighbors.
def predict(neighbors):
    predictt = None
    predictt = np.mean(neighbors)
    return predictt

# Function evaluation: to evaluate model using RMSE
# Parameters:
#   actual_cls, and prediction class
def evaluation(actual_cls, predictions):
    actual_cls_size = len(actual_cls)
    squared_error_array = 0.0
    for x in range(0,actual_cls_size):
        squared_error = (abs(actual_cls[x]-predictions[x]))/10
        squared_error_array += (squared_error**2)
    mse = squared_error_array/float(len(actual_cls))
    return np.sqrt(mse)
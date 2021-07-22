# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:57:06 2021

@author: SYAVIRA TZ
"""
import numpy as np
from KnnRegression import Load,fit, predict, evaluation

trainingSet=[]
testSet=[]
actual_ = []
split = 0.67  
datas = Load ('image_segmentation.txt', split, trainingSet, testSet)
print ('Train set: ' + repr(len(trainingSet)))
print ('Test set: ' + repr(len(testSet)))
k = 5
list_neigh = []
predictions=[]
for row in range(len(testSet)):
    actual_class_val = testSet[row][-2]
    actual_.append(actual_class_val)
    knn_test = fit (trainingSet, testSet[row], k)
    prediction = predict(knn_test)
    predictions.append(prediction)
RMSE = evaluation(actual_, predictions)
print('RMSE: ' + str(RMSE))
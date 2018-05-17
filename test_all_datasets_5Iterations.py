# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:16:39 2018

@author: Eduardo Fidalgo Fernández (efidf@unileon.es)
Based on the code from: https://gogul09.github.io/software/flower-recognition-deep-learning

Script to run all the tests over the five datasets of the paper for PRL.
Use the VGG16 extracted features or MobileNet into an SVM classifier.

The script can run directly over the Soccer dataset samples:
- Soccer: original images
- SoccerAutoBlurBB: original images after apply AutoBlur filtering technique and cropped them with the corresponding Bounding Box

"""
# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn import svm
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import statistics as st
import effTools as eff
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# List of the datasets to be analyzed
#datasets_available = ["Soccer", "SoccerAutoBlurBB",
#           "17flowers",  "17flowersAutoBlurBB",
#		   "Birds",  "BirdsAutoBlurBB"]

datasets_available = ["Soccer", "SoccerAutoBlurBB"]

#conf_file = datasets_available.index("ImageNet_7Arthropods")
							     
# Select the configuration file available
#file = 'conf_' + datasets_available[conf_file] + '.json'

# load the user configs. It has been created a json file per dataset
#config = eff.load_json_file(file)


# datasets to be analyzed
dataset = datasets_available

#test_set = [0.375, 0.375, 0.375, 
#            0.25, 0.25, 0.25, 
#            0.25, 0.25, 0.25]

test_set = [0.25, 0.25, 0.25]


for ds in range(0, len(dataset)):

    # Run 5 times the experimentation and compute the average
    avg_accuracy = []
    avg_recall = []

    for exp in list(range(1,6)):
        print("Running experiment " +  str(exp) + " for dataset ")    
        # config variables
        model_name    = "mobilenet"
        weights       = "imagenet"
        include_top   = 0 
        train_path    = "datasets/" + dataset[ds]
        features_path = "output/" + dataset[ds] + "/" + model_name + "/features.h5" 
        labels_path   = "output/" + dataset[ds] + "/" + model_name + "/labels.h5"
        test_size     = test_set[ds]
        results       = "output/" + dataset[ds] + "/" + model_name + "/results" + dataset[ds] + ".txt"
        model_path    = "output/" + dataset[ds] + "/" + model_name + "/model"
        classifier_path = "output/" + dataset[ds] + "/" + model_name + "/classifier.pickle"
        
        # import features and labels
        h5f_data  = h5py.File(features_path, 'r')
        h5f_label = h5py.File(labels_path, 'r')
        
        features_string = h5f_data['dataset_1']
        labels_string   = h5f_label['dataset_1']
        
        features = np.array(features_string)
        labels   = np.array(labels_string)
        
        h5f_data.close()
        h5f_label.close()


        # verify the shape of features and labels
        print ("[INFO] features shape: {}".format(features.shape))
        print ("[INFO] labels shape: {}".format(labels.shape))
        
        print ("[INFO] training started...")
        # split the training and testing data
        (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                          np.array(labels),
                                                                          test_size=test_size,
                                                                          random_state=exp)
        
        print ("[INFO] splitted train and test data...")
        print ("[INFO] train data  : {}".format(trainData.shape))
        print ("[INFO] test data   : {}".format(testData.shape))
        print ("[INFO] train labels: {}".format(trainLabels.shape))
        print ("[INFO] test labels : {}".format(testLabels.shape))
        
        # use logistic regression as the model
        print ("[INFO] creating model...")
        #model = LogisticRegression(random_state=seed)
        model = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
        model.fit(trainData, trainLabels)
        
        # use rank-1 and rank-5 predictions
        print ("[INFO] evaluating model...")
        
        # Run it five times and compute average of the recall, to match with the 
        # results computed in MATLAB
        
        rank_1 = 0
        rank_5 = 0
        
        # loop over test data
        for (label, features) in zip(testLabels, testData):
          # predict the probability of each class label and
          # take the top-5 class labels
          predictions = model.predict_proba(np.atleast_2d(features))[0]
          predictions = np.argsort(predictions)[::-1][:5]
        
          # rank-1 prediction increment
          if label == predictions[0]:
            rank_1 += 1
        
          # rank-5 prediction increment
          if label in predictions:
            rank_5 += 1
        
        # convert accuracies to percentages
        rank_1 = (rank_1 / float(len(testLabels))) * 100
        rank_5 = (rank_5 / float(len(testLabels))) * 100
        
        # write the accuracies to file
        # f.write("Rank-1: {:.2f}%\n".format(rank_1))
        # f.write("Rank-5: {:.2f}%\n\n".format(rank_5))
        
        # evaluate the model of test data
        preds = model.predict(testData)
        
        accuracy = sklearn.metrics.accuracy_score(testLabels, preds)
        accuracy = accuracy * 100
        avg_accuracy.append(accuracy)
        
        recall = sklearn.metrics.recall_score(testLabels, preds, average='macro')
        recall = recall * 100
        avg_recall.append(recall)
        
    f = open(results, "w")
    avg_acc = st.mean(avg_accuracy)
    std_acc= st.stdev(avg_accuracy)
    avg_rec = st.mean(avg_recall)
    std_rec= st.stdev(avg_recall)
    # write the classification report to file
    f.write("Averaged Accuracy: {:.2f}%\n\n".format(avg_acc))
    f.write("Std deviation (Accuracy): {:.2f}%\n\n".format(std_acc))
    f.write("Averaged Recall: {:.2f}%\n\n".format(avg_rec))
    f.write("Std deviation (Recall): {:.2f}%\n\n".format(std_rec))

    # write the classification report to file
    #f.write("{}\n".format(classification_report(testLabels, preds)))
    f.close()

    # dump classifier to file
    #print ("[INFO] saving model...")
    #pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
# print ("[INFO] confusion matrix")

# get the list of training lables
# labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
# cm = confusion_matrix(testLabels, preds)
# sns.heatmap(cm,
#             annot=True,
#             cmap="Set2")
# plt.show()
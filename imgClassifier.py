# imgClassifier.py
# Created by T. J. Tkacik for Assignment 5 of COMP 590
# Spring of 2014 at the University of North Carolina
import sys, time, os, math, random, Image, svm, numpy as np
from svm import *
from svmutil import *
from libsvm import *

class imgClassifier(object):
    def __init__(self, folder = "images", loud = False):
        self.loud = loud
        self.folder = folder
        self.f = 5
        classes = {}
        models = {}
        predictions = {}
        evaluations = {}
        
        #Computing Features
        for root, subFolders, files in os.walk(self.folder):
            for folder in subFolders:
                classes[folder] = []
                for image in os.listdir(os.path.join(root,folder)):
                    if not image.startswith("._"):
                        image = Image.open(os.path.join(root,folder,image))
                        image = image.resize((32,32), Image.ANTIALIAS)
                        image = np.array(image).reshape(-1).tolist()
                        classes[folder].append(image)
                random.shuffle(classes[folder])
         
        #Training
        for key in classes.keys():
            folds = [[] for x in range(self.f)]
            for i in range(len(classes[key])):
                folds[i%self.f].append(classes[key][i])
            classes[key] = folds

        for key in classes.keys():
            models[key] = []
            if os.path.isfile(key+"_lin.model") and os.path.isfile(key+"_rbf.model"):
                models[key].append(svm_load_model(key+"_lin.model"))
                models[key].append(svm_load_model(key+"_rbf.model"))
            else:
                trainlabels, trainData = [], []
                testlabels, testData = [], []
                extlabels, extData = [], []
                for i in range(self.f-2):
                    positives = classes[key][i]
                    trainlabels += [1]*len(positives)
                    trainData += positives
                    for otherKey in classes.keys():
                        if otherKey != key:
                            negatives = classes[otherKey][i]
                            trainlabels += [0]*len(negatives)
                            trainData += negatives
                positives = classes[key][self.f-2]
                testlabels += [1]*len(positives)
                testData += positives
                for otherKey in classes.keys():
                    if otherKey != key:
                        negatives = classes[otherKey][self.f-2]
                        testlabels += [0]*len(negatives)
                        testData += negatives
                positives = classes[key][self.f-1]
                extlabels += [1]*len(positives)
                extData += positives
                for otherKey in classes.keys():
                    if otherKey != key:
                        negatives = classes[otherKey][self.f-1]
                        extlabels += [0]*len(negatives)
                        extData += negatives
                        
                pb = svm_problem(trainlabels, trainData)
    
                c = 1.0
                g = 1.0/3072
                bestClin, bestCrbf, bestG = c, c, g
                acclin, accrbf = 0, 0
                evaluations[key] = ([], [])
                if self.loud: print "Training Models for ", key
                for i in range(-4,5):
                    newC = c*(10**(i*3))
                    pmlin = svm_parameter("-q -t 0 -c "+ str(newC))
                    mdlin = svm_train(pb, pmlin)
                    label, acc, val = svm_predict(testlabels, testData, mdlin, '-q')
                    print newC, acc[0]
                    #evaluations[key][0].append((newC, acc[0]))
                    if acc[0] > acclin:
                        acclin = acc[0]
                        bestClin = newC
                for i in range(4):
                    newC = c*(10**((i-3)*3))
                    #evaluations[key][1].append((newC, []))
                    for j in range(-3,2):
                        newG = g*(10**(j*3))
                        pmrbf = svm_parameter("-q -t 1 -m 800 -c "+ str(newC) + " -g "+ str(newG))
                        mdrbf = svm_train(pb, pmrbf)
                        label, acc, val = svm_predict(testlabels, testData, mdrbf, '-q')
                        #print newC, newG, acc[0]
                        #evaluations[key][1][i][1].append((newG, acc[0]))
                        if acc[0] > accrbf:
                            accrbf = acc[0]
                            bestCrbf, bestG = newC, newG
                if self.loud: print "Best linear Cost, best RBF Cost, best Gamma:"
                if self.loud: print bestClin, bestCrbf, bestG
                
                pmlin = svm_parameter("-b 1 -q -t 0 -c "+ str(bestClin))
                pmrbf = svm_parameter("-b 1 -q -t 1 -c "+ str(bestCrbf)+ " -g "+ str(bestG))
                pb = svm_problem(trainlabels+testlabels, trainData+testData)
                mdlin = svm_train(pb, pmlin)
                mdrbf = svm_train(pb, pmrbf)
                svm_save_model(key+"_lin.model", mdlin)
                svm_save_model(key+"_rbf.model", mdrbf)
                models[key].append(mdlin)
                models[key].append(mdrbf)
                
        #Testing
        confmatrix = np.zeros((len(classes.keys()),len(classes.keys())))
        for k in range(len(classes.keys())):
            key = classes.keys()[k]
            for img in classes[key][self.f-1]:
                argmax = -1
                probmax = 0
                for l in range(len(models.keys())):
                    otherKey = classes.keys()[l]
                    prob = svm_predict([0], [img], models[otherKey][0], '-b 1 -q')[2][0][0]
                    if prob > probmax:
                        probmax, argmax = prob, l
                if argmax == -1:
                    print "ERROR"
                else: confmatrix[k][argmax] += 1
        for row in confmatrix:
            print row
            
        confmatrix = np.zeros((len(classes.keys()),len(classes.keys())))
        for k in range(len(classes.keys())):
            key = classes.keys()[k]
            for img in classes[key][self.f-1]:
                argmax = -1
                probmax = 0
                for l in range(len(models.keys())):
                    otherKey = classes.keys()[l]
                    prob = svm_predict([0], [img], models[otherKey][1], '-b 1 -q')[2][0][0]
                    if prob > probmax:
                        probmax, argmax = prob, l
                if argmax == -1:
                    print "ERROR"
                else: confmatrix[k][argmax] += 1
        for i in range(len(classes.keys())):
            print classes.keys()[i], "\t", confmatrix[i]
            
        """for key in evaluations.keys():
            print key
            lin = evaluations[key][0]
            for x,y in lin:
                print x, y
            
            rbf = evaluations[key][1]
            for x,y in rbf:
                print x, y"""
   
if  __name__ =='__main__':
    loud = False
    folder = "images"
    
    if "--help" in sys.argv:
        print """
        imgClassifier.py by T. J. Tkacik
        
        Accepted flags:

        --help    for this help information
        -l        for loud output, default False
        -f        to select folder, default 'images'

        Example:   imgClassifier.py -l -f images
        
        Note:   imgClassifier.py is prone to fail by resulting in a
                segmentation fault (core dump). Because of this, models
                that are successfully generated are saved as class_kernal.model
                files and used in later runs. To retrain the parameters of an 
                SVM, delete the model files from this directory.
        """
        sys.exit(0)
    
    if "-l" in sys.argv:
        loud = True
    if "-f" in sys.argv:
        folder = sys.argv[sys.argv.index("-f")+1]

    classifier = imgClassifier(folder, loud)
      
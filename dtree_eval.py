'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt
#import os

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'Questions\CIS419-master\Assignment1\hw1_skeleton\data\SPECTF.dat'
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    # shuffle the data
    
    
    # split the data
    # 10-fold splitting.
    Xsize = len(X[:,:])
    # print(Xsize)
    # Xsize work, dividing data into N instances:
    DataInstance = 10
    DatPerInstance = Xsize//DataInstance

    #Honestly I am not used to Python. Just decalre global variable and get away with it.
    LocalIDAccuracy = []
    StumpAccuracy = []
    TripleAccuracy = []
    

    for trial in range(99):
        #Shuffle the data
        idx = np.arange(n)
        np.random.seed(trial)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        #print(X)


        #Split data into N arrays:
        '''
        for divide_count in range(DataInstance - 1):
            currentCount = divide_count*DatPerInstance
            nextCount = currentCount + DatPerInstance
            XEleArray = X[currentCount:nextCount,:]
            YEleArray = y[currentCount:nextCount,:]
            XDataArray.append(XEleArray)
            YDataArray.append(YEleArray)
        # Now we will make training and testing.
        
        '''
        #Split data return a 3D array -> pre-split caan't be used. Let's do something else
        for train_count in range (DataInstance - 1):
            #Prepare training variable
            testMin = train_count*DatPerInstance
            testMax = testMin + DatPerInstance
            #Create training set
            Xtest = X[testMin:testMax,:]
            ytest = y[testMin:testMax,:]
            #Creating training set
            if testMin == 0:
                Xtrain = X[testMax:,:]
                ytrain = y[testMax:,:]
            else:
                xtrainleft = np.array(X[0:testMin,:])
                xtrainright = np.array(X[testMax:,:])
                ytrainleft = np.array(y[0:testMin,:])
                ytrainright = np.array(y[testMax:,:])
                Xtrain = np.concatenate((xtrainleft,xtrainright))
                ytrain = np.concatenate((ytrainleft,ytrainright))
            #Now the data combine is complete, let's train
            clf = tree.DecisionTreeClassifier()
            stump = tree.DecisionTreeClassifier(max_depth= 1)
            tripletree = tree.DecisionTreeClassifier(max_depth = 3)
            #train them side by side. What can go wrong?
            clf.fit(Xtrain,ytrain)
            stump.fit(Xtrain,ytrain)
            tripletree.fit(Xtrain,ytrain)
            #Train complete. Testing
            y_result = clf.predict(Xtest)
            y_result_1d = stump.predict(Xtest)
            y_result_3d = tripletree.predict(Xtest)
            #Cheking
            accuracy_check = accuracy_score(ytest,y_result)
            accuracy_1d = accuracy_score(ytest,y_result_1d)
            accuracy_3d = accuracy_score(ytest,y_result_3d)
            #print(accuracy_check)
            LocalIDAccuracy.append(accuracy_check)
            StumpAccuracy.append(accuracy_1d)
            TripleAccuracy.append(accuracy_3d)
            
            #Clear data.

    # Done, 




   
    
    
    # TODO: update these statistics based on the results of your experiment
    meanDecisionTreeAccuracy = np.mean(LocalIDAccuracy)
    stddevDecisionTreeAccuracy = np.std(LocalIDAccuracy)
    meanDecisionStumpAccuracy = np.mean(StumpAccuracy)
    stddevDecisionStumpAccuracy = np.std(StumpAccuracy)
    meanDT3Accuracy = np.mean(TripleAccuracy)
    stddevDT3Accuracy = np.std(TripleAccuracy)

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print ("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    print ("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    print ("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.

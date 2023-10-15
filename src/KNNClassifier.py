import os
import matplotlib.pyplot as plt
import numpy as np
import math as m
import random
from collections import Counter
from data.data_all import adult_data, ionosphere_data, diabetes_data, spambase_data
import time

class KNNClassifier:
    def __init__(self,dataSet,col1,col2,divisor):
        #specify which columns in the dataset to use
        self.col1 = col1
        self.col2 = col2
        #for larger datasets this is used to cut the data smaller and random shuffling will be used
        #cutting the data down on large data sets could be useful to speed up the algorithm
        self.divisor = divisor
       
        self.classArray = self.getClassificationsArray(dataSet)
       
        self.data = self.convertDataToTupleArray(dataSet)
        self.kAcc = []


    def _getDistance(self,first:list, second:list):
       
        return m.sqrt((float(second[0]) - float(first[0])) ** 2 + (float(second[1]) - float(first[1])) ** 2)
    
    def classifyNewPoint(self,dataSet: list,newPoint: list,k:int,classes: list):

        distances = []
        for points in dataSet:
            
            
            for tuple in points:
                distance = self._getDistance(newPoint,tuple[0])
               
                distances.append((distance,tuple[1]))
        
        distances = sorted(distances,key=lambda pair:pair[0])
        outputfreq = Counter()
        for neighbor in distances[:k]:
            outputfreq[neighbor[1]] +=1

        return max(outputfreq, key = outputfreq.get)
    

    def convertDataToTupleArray(self,dataSet: list):
        
        #This will split the dataset to a self.divisor of its size
        #But still implements randomness to spread data so it wont have biases
        array1 = []
        array2= []
        classification = []
        if self.divisor > 1:
            randomArray = list(range(0,len(dataSet)))
            random.shuffle(randomArray)
            randomArray = randomArray[0:len(randomArray)//self.divisor]

            
            for i in randomArray:
                array1.append(dataSet[:,self.col1][i])
                array2.append(dataSet[:,self.col2][i])
                classification.append(dataSet[:,-1][i])
        else:
            
            array1 = dataSet[:,self.col1]
            array2 = dataSet[:,self.col2]
            classification = dataSet[:,-1]
        
        finalArray = []
        for i in range(len(classification)):
            finalArray.append(([array1[i],array2[i]],classification[i]))
        
        return finalArray
    
    def getClassificationsArray(self,nparray: np.ndarray):
        return np.unique(nparray[:,-1])
    
    def findBestK(self):
        
        j,m = divmod(len(self.data),5)
        arraysub5 = list(self.data[i*j+min(i, m):(i+1)*j+min(i+1, m)] for i in range(5))
        classes = self.classArray
        length = len(self.data)

        lengthOfClasses = len(classes)
        kAccuracy = []
        for k in range(1,21):
            print(f"K: {k}")
            #using 5 Fold cross validation to find best K
            correct = 0
            for i in range(5):

                #define testing and training sets
                print(f"iteration: {i}")
                testingSet = arraysub5[i]
                trainingSet = []
                for j in range(5):
                    if j != i:
                        trainingSet.append(arraysub5[j])
                
                #iterate through testing set and classify the point amongst the other training data
                for point in testingSet:
                    guess = self.classifyNewPoint(trainingSet,point[0],k,classes)
                    
                    if guess == point[1]:
                        correct +=1
                print(f"Correct Predictions: {correct}")
            kAccuracy.append(correct/length)
            print(kAccuracy)
        
        print(f"Acurracy: {correct/length}")    
        self.kAcc = kAccuracy
            
    def showKplot(self):
        if len(self.kAcc) == 0:
            print("no K accuracy data: Run K function first ")
            return
        x = [i for i in range(1,21)]
        plt.title(f"K accuracy data with columns {self.col1+1} and {self.col2+1}")
        plt.scatter(x,self.kAcc)
        plt.plot(x,self.kAcc)
        plt.xticks(x,x)
        plt.show()
        return

    def saveKplot(self, filename):
        if len(self.kAcc) == 0:
            print("no K accuracy data: Run K function first ")
            return
        plt.clf()
        x = [i for i in range(1,21)]
        plt.title(f"K accuracy data with columns {self.col1+1} and {self.col2+1}")
        plt.scatter(x,self.kAcc)
        plt.plot(x,self.kAcc)
        plt.xticks(x,x)
        save_path = f'/Users/levivanv/Downloads/{filename}'
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

num = 1
_, cols = diabetes_data.shape
for i in range(cols):
  for j in range(i+1, cols):
    start = time.time()
    temp = KNNClassifier(diabetes_data, i, j, 1)
    temp.findBestK()
    end = time.time()
    filename = f'Diabetes_{i}_{j}.png'
    temp.saveKplot(filename)
    

## UNCOMMENT FROM HERE TO NEXT END FOR ADULT DATA RESULTS
# adultData = np.loadtxt("data\\adult\\adult.data", delimiter=',', dtype='str')
# arr = np.array(adultData)

# # remove rows with missing values
# arr = arr[~np.any(arr == " ?", axis=1)]

# # remove duplicate rows
# arr = np.unique(arr, axis=0)

# start = time.time()
# temp = KNNClassifier(arr,0,2,8)
# temp.findBestK()
# end = time.time()
# temp.showKplot()

# print(f"Time taken to complete training of adult data: {end -start}")

########

### UNCOMMENT FOR IONO DATA
# ionoData = np.loadtxt("data\ionosphere\ionosphere.data", delimiter=',', dtype='str')
# arr = np.array(ionoData)

# # remove rows with missing values
# arr = arr[~np.any(arr == " ?", axis=1)]

# # remove duplicate rows
# arr = np.unique(arr, axis=0)

# _,cols = arr.shape
# for i in range(cols):
#     for j in range(i+1,cols):
#         temp = KNNClassifier(arr,i,j,1)
#         temp.findBestK()
#         temp.showKplot()
        
######

### LUNG CANCER DATA

# lungCancerData = np.loadtxt("data\lung_cancer\lung_cancer.data", delimiter=',', dtype='str')
# arr = np.array(lungCancerData)

# # remove rows with missing values
# arr = arr[~np.any(arr == " ?", axis=1)]

# # remove duplicate rows
# arr = np.unique(arr, axis=0)
# _,cols = arr.shape
# num = 1

# for i in range(cols):
#     for j in range(i+1,cols):
#         temp = KNNClassifier(arr,i,j,1)
#         temp.findBestK()
#         temp.showKplot()
#         num +=1


#####

# wineData = np.loadtxt("data\wine\wine.data", delimiter=',', dtype='str')
# arr = np.array(wineData)

# # remove rows with missing values
# arr = arr[~np.any(arr == " ?", axis=1)]

# # remove duplicate rows
# arr = np.unique(arr, axis=0)
# _,cols = arr.shape
# num = 1

# for i in range(cols):
#     for j in range(i+1,cols):
#         temp = KNNClassifier(arr,i,j,1)
#         temp.findBestK()
#         temp.showKplot()
#         num +=1
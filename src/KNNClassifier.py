
from ucimlrepo import fetch_ucirepo 
import pandas
import matplotlib.pyplot as plt
import numpy as np

# fetch dataset 
# rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
# print(rice_cammeo_and_osmancik.data.features['Area'])

# y = np.array(rice_cammeo_and_osmancik.data.features['Area'] )
# y = np.array(rice_cammeo_and_osmancik.data.targets )

# # metadata 
# print(rice_cammeo_and_osmancik.metadata) 
  
# # variable information 
# print(rice_cammeo_and_osmancik.variables) 


import math as m
import random

class KNNClassifier:
    
    

    def __init__(self,dataSet,k):
        self.data = dataSet
        self.kNumbers = k
        #2D array where the first point in each row is the centre for that cluster
        self.clusters = []
        
    def getDistance(self,first, second):
        return m.sqrt( (second[0]-first[0])**2 + (second[1]-first[1])**2 )
    
    def assignCentres(self):
        
        centres = []
        
        #wipe clusters to reset
        for row in self.clusters:
            centres.append(row[0])
        self.clusters = []
        for i in centres:
            self.clusters.append([i])

        for i in self.data:
            
            #find closest cluster centre to point
            closestCentre = centres[0]
            iter = 0
            pos = 0
            
            for centre in centres:
                
                if self.getDistance(closestCentre,i) >= self.getDistance(centre,i):
                    closestCentre = centre
                    pos = iter
                iter +=1
            
            #add the point to the found closest point
            print("before",self.clusters)
            self.clusters[pos].append(i)
            print("after",self.clusters)
            # for row in self.clusters:
            #     if row[0] == closestCentre:
            #         print("triggers")
            #         row.append(i)
            

        return
    
    def newCentres(self):

        if len(self.clusters) == 0:
            print("Cannot assign centres to an empty cluster array.")
            return

        for row in self.clusters:
            xAverage = 0.0
            yAverage = 0.0
            
            for point in row[1:]:
                xAverage += point[0]
                yAverage += point[1]

            if len(row[1:]) != 0:
                xAverage = xAverage/len(row[1:])
                yAverage = yAverage/len(row[1:])
            
            row[0][0] = xAverage
            row[0][1] = yAverage
    
    def runKNN(self,maxX,maxY):

        for i in range(0,self.kNumbers):
            initx = random.randint(0,maxX)
            inity = random.randint(0,maxY)
            self.clusters.append([[initx,inity]])
            print("initial centres")
            print(self.clusters)

        for i in range(10):
            print("iteration ",i)
            self.assignCentres()
            print("Assigned centres with points\n")
            print("centres: ",self.clusters)

            self.newCentres()
            print("New Centres:",end="")
            for row in self.clusters:
                print(" ",row[0],end="")
            print()

data = [
[5,2],
[2,4],
[9,5],
[4,6],
[5,2],
[1,5],
[6,7],
[4,2],
[6,4],
[9,2],
[4,5],
[1,6],
[4,7],
[3,6],
[1,1],
[8,4],
[8,7],
[7,2],
[2,2],
[2,1],
[1,2],
[1,4],
[2,6],
[7,7],
[7,4],
[3,4],
[1,4]
]
x = [i[0] for i in data]
y = [i[1] for i in data]


example = KNNClassifier(data,2)

example.runKNN(8,9)
plt.scatter(x,y)
plt.show()
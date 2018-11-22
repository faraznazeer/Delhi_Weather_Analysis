import csv
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt 
import time

print("\n Packages Loaded  ")

temp = []
dates = []

def get_data(filename):
    
    n = 0
    c = 1
    avg = 0
    with open(filename, "r") as file:
        Reader = csv.reader(file)
        next(Reader)
        for data in Reader:
            if data[11] != "":
                if(c > 0 ):
                    avg = avg + float(data[11])
                    current_date = data[0]
                else:
                    dates.append(  current_date  )
                    temp.append(  float ( avg/24 )  )
                    n = n + 1
                    avg = 0
                c = (c + 1)%24

        dates.append(  current_date  )
        temp.append(  float ( avg/c )  )
        n = n + 1

    index = np.reshape( range(n), (n,1) )

    return n, index 


def TrainModel(index, temp):

    rbf = SVR( kernel="rbf", C=1e3, gamma = 0.01)
    rbf.fit(index,temp)
    return rbf


def PlotTestDataPrediction(model, index, temp):
    #print(test_temp)
    plt.scatter(index , temp, color = "black", label = "Data")
    plt.plot( index , model.predict(index), color = "blue", label = "RBF")
    plt.xlabel("Dates (Data Point)")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()



def predict(model, datapoint):
    return model.predict(datapoint)[0]



print("\n Loading Data.........")
n, index = get_data("Delhi_Weather_Data.csv")
print("\n Data Successfully Loaded !!!")


print("\n Training Begins...........")
start = time.time()

model = TrainModel(index, temp)

duration = time.time() - start 
print("\n Training Completed in " + str(duration)+ " Seconds" )


PlotTestDataPrediction(model, index, temp)

print( "\n Temperature at next data point " + str(n+1) +" : "+str(predict( model, n + 1)) )

datapoint = input("\n Enter the data point to predict : " )

print( "\n Temperature at data point " +str(datapoint)+ ": " + str(predict( model, datapoint)) )



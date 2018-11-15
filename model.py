import csv
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

index = list(range(2002))
index = np.reshape( index , (2002, 1) )
dates = []
temp = []

def get_data(filename):
    
    with open(filename, "r") as file:
        Reader = csv.reader(file)
        next(Reader)
        c = 0
        d=0
        for data in Reader:
            if data[11] != "" and c == 0:
                dates.append(  data[0]  )
                temp.append(  float ( data[11] )  )
                d = d+1
            c = (c+1)%50
    print(d)



def predict_temp(dates, price, x):

    
    #lin = SVR( kernel="linear", C=1)
    #poly = SVR( kernel="poly", C=1, degree=2)
    rbf = SVR( kernel="rbf", C=1000, gamma=0.1)

   # lin.fit(index,temp)    
    #poly.fit(index,temp)
    rbf.fit(index,temp)

    plt.scatter(index, temp, color = "black", label = "Data")
    #plt.plot( index, lin.predict(index), color = "red", label = "Linear")
   # plt.plot( index, poly.predict(temp), color = "green", label = "Polynomial")
    plt.plot( index, rbf.predict(index), color = "blue", label = "RBF")

    plt.xlabel("Dates")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

    #return  lin.predict(x)[0], poly.predict(x)[0], rbf.predict(x)[0]


get_data("Delhi_Weather_Data.csv")

print( predict_temp(dates,temp,100317) )

#print ( get_accuracy() )
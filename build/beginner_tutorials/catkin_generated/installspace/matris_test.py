import pandas as pd
import numpy as np

def main():

    #VARIABLES
    w1_array = pd.read_csv('data/w1_array.csv')
    w2_array = pd.read_csv('data/w2_array.csv')
    w3_array = pd.read_csv('data/w3_array.csv')
    wo_array = pd.read_csv('data/wo_array.csv')
    bias1 = pd.read_csv('data/bias1.csv')
    bias2 = pd.read_csv('data/bias2.csv')
    #bias2 = np.array(bias2)
    print(bias2)
    
    def activ_func( inp): #leaky relU aktivation function 
        if inp < 0:
            return 0.01 * inp
        else:
            return inp

    def ileriYay( newDataFrame, bias1_array, bias2_array): # forward propagation 
        #hidden layer 1
        hid = newDataFrame.dot(w1_array) # multiply input values with weights for first hidden layer
        hid1 = hid + bias1_array # add bias values
        x = len(hid1)
        id = 0
        matris1 = np.full((x, 1), None)
        matris2 = np.full((x, 1), None)
        matris3 = np.full((x, 1), None)
        matris = np.full((x, 1), None)

        for row in hid1:
            matris1[id] = float(activ_func(row)) # put the values in activation function
            id = id + 1
        #hidden layer 2
        id = 0
        hid = newDataFrame.dot(w2_array)  # multiply input values with weights for second hidden layer
        hid2 = hid + bias1_array # add bias values
        for row in hid2:
            matris2[id] = float(activ_func(row)) # put the values in activation function
            id = id + 1
        #hidden layer 3
        id = 0
        hid = newDataFrame.dot(w3_array) # multiply input values with weights for third hidden layer
        hid3 = hid + bias1_array # add bias values
        for row in hid3:
            matris3[id] = float(activ_func(row)) # put the values in activation function
            id = id + 1
        #out layer
        id = 0
        #combining matrices
        com_matris = np.hstack((matris1, matris2, matris3))
        out = com_matris.dot(wo_array) # multiply three hidden layer values with weights for output
        out1 = out + bias2_array # add bias values

        for row in out1:
            for num in row:
                matris[id] = float(activ_func(num))  # put the values in activation function
            id = id + 1
        return matris
    
    #FuNCTIONS

    def detect_flower(out1): # detecting flower type if it is between defined values
        id = 0
        for row in out1:

            if  -1 < row < 0.7: 
                print(f"{id}. flower is Iris-setosa",row, input_ndata[id][5])

            elif 0.7 < row < 1.6:
                print(f"{id}. flower is Iris-versicolor",row, input_ndata[id][5])

            elif  1.6 < row < 5:  
                print(f"{id}. flower is Iris-virginica",row, input_ndata[id][5])

            else:
                print(f"{id}. cannot detect",row, input_ndata[id][5])

            id = id + 1
   
    #GET  INPUT VALUES FOR TESTING
    
    out1 = ileriYay(input_data, bias1, bias2)
    detect_flower(out1)
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    main()


import numpy as np # library for math operations 
import pandas as pd # library used for pulling required dataframes from files
import random   # used for generating random values
from sklearn.model_selection import train_test_split #used for splitting dataset as test and train 
import matplotlib.pyplot as plt # library used for error showing graph
import csv  # used for writing and reading values from csv file

def matris_train():
    # initial values
    # give weights and bias random values between 0.1 and 0.3
    # there are four inputs 3 hidden layers and 1 ouput 
    # so we have 15 weight values and 2 biases
    w1 = random.uniform(0.1,0.3)
    w2 = random.uniform(0.1,0.3)
    w3 = random.uniform(0.1,0.3)
    w4 = random.uniform(0.1,0.3)
    w5 = random.uniform(0.1,0.3)
    w6 = random.uniform(0.1,0.3)
    w7 = random.uniform(0.1,0.3)
    w8 = random.uniform(0.1,0.3)
    w9 = random.uniform(0.1,0.3)
    w10 = random.uniform(0.1,0.3)
    w11 = random.uniform(0.1,0.3)
    w12 = random.uniform(0.1,0.3)
    w13 = random.uniform(0.1,0.3)
    w14 = random.uniform(0.1,0.3)
    w15 = random.uniform(0.1,0.3)
    bias1 = random.uniform(0.1,0.3)
    bias2 = random.uniform(0.1,0.3)

    # categorizing weights into four matrices
    w1_array = np.array([[w1],[w2],[w3],[w4]]) # this goes the first hidden layer
    w2_array = np.array([[w5],[w6],[w7],[w8]]) # this goes second hidden layer     
    w3_array = np.array([[w9],[w10],[w11],[w12]]) # this goes third hidden layer
    wo_array = np.array([[w13],[w14],[w15]]) # this goes to output 
    
    # CONSTANTS
    LR = np.array([0.0001]) # learning rate 
    exp_out1 = np.array([[0]])  #Iris-setosa asigned 0 
    exp_out2 = np.array([[1]])  #Iris-versicolor asigned 1
    exp_out3 = np.array([[2]])  #Iris-virginia asigned 2

    def test_train(): # splits dataset as test(%20) and train(%80) 
        dataFrame = pd.read_csv("data/Iris.csv", index_col ="Id")
        dataFrame = np.array(dataFrame)

        train, test = train_test_split(dataFrame, test_size=0.2)

        df = pd.DataFrame(train)
        df.to_csv("data/train_dataset.csv") # writing train dataframe to file 

        df = pd.DataFrame(test)
        df.to_csv("data/test_dataset.csv") # writing test dataframe to file

    def activ_func( inp): #leaky relU aktivation function 
        if inp < 0:
            return 0.01 * inp
        else:
            return inp

    def hatahesaplama( out_matris, dataFrame): # calculating error values of output
        for row in dataFrame:
            id = 0
            flower = row[5]

            if flower == "Iris-setosa":
                exp_out = exp_out1
            elif flower == "Iris-versicolor":
                exp_out = exp_out2
            elif flower == "Iris-virginica":
                exp_out = exp_out3

            err = 1/2 * pow((out_matris[id] - exp_out),2)
            id = id + 1

        return err

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

        return matris, com_matris

    def geriYay(out_matris, com_matris, dataFrame, w1_array, w2_array, w3_array, wo_array, newDataFrame, bias1, bias2):
        # back propagation

        id = 0
        for row in dataFrame:

            flower = row[5]

            if flower == "Iris-setosa":
                exp_out = exp_out1
            elif flower == "Iris-versicolor":
                exp_out = exp_out2
            elif flower == "Iris-virginica":
                exp_out = exp_out3

            if out_matris[id] >= 0:
                num = 1
            elif out_matris[id] < 0:
               num = 0.01
            # changing weight and bias values with back propagation
            turev1 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w13 * num * num # derivative
            w1_array = w1_array - np.transpose(LR * turev1) 

            turev2 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w14 * num * num # derivative
            w2_array = w2_array - np.transpose(LR * turev2) 

            turev3 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w15 * num * num # derivative
            w3_array = w3_array - np.transpose(LR * turev3) 

            turevout = (out_matris[id] - exp_out) * com_matris[id] * num # derivative
            wo_array = wo_array - np.transpose(LR * turevout) 

            turbias1 = (out_matris[id] - exp_out) * w13 * num  * num + (out_matris[id] - exp_out) * w14 * num  * num + (out_matris[id] - exp_out) * w15 * num  * num # derivative
            bias1 = bias1 - LR * turbias1

            turbias2 = (out_matris[id] - exp_out) * num * 3 # derivative
            bias2 = bias2 - LR * turbias2

            id = id + 1

        return w1_array, w2_array, w3_array, wo_array, bias1, bias2

    test_train() #split dataset as test(%20) and train(%80) 
    err_array = [] #create an array for error values

    #INPUT
    # getting train dataset values for train on them
    dataFrame = pd.read_csv("data/train_dataset.csv")
    dataFrame = np.array(dataFrame)
    newDataFrame = dataFrame[:, 1:5]
    x = len(dataFrame)
    itr = 1

    # train dataset for 1000 iterations
    while(itr < 1000):
        bias1_array = np.full((x, 1), bias1)
        bias2_array = np.full((x, 1), bias2)
        out_matris, com_matris = ileriYay(newDataFrame, bias1_array, bias2_array)
        err = hatahesaplama(out_matris, dataFrame)
        err_array = np.append(err_array, err)
        w1_array, w2_array, w3_array, wo_array, bias1, bias2 = geriYay(out_matris, com_matris, dataFrame, w1_array, w2_array, w3_array, wo_array, newDataFrame, bias1, bias2)
        print(itr)
        itr = itr + 1

    #show error
    err_array = np.array(err_array)
    plt.plot(err_array)
    plt.ylabel('error')
    plt.show()
    
   

    # convert each element of the array to float
    w1_array = [float(row) for row in w1_array]

    # open the file in write mode
    with open('data/w1_array.csv', 'w', newline='') as f:
        #create the csv writer
        writer = csv.writer(f)
        # write the values as a row to the csv file
        writer.writerow(w1_array)


    # convert each element of the array to float
    w2_array = [float(row) for row in w2_array]
    
    # open the file in write mode
    with open('data/w2_array.csv', 'w', newline='') as f:
        #create the csv writer
        writer = csv.writer(f)
        # write the values as a row to the csv file
        writer.writerow(w2_array)
    
    # convert each element of the array to float
    w3_array = [float(row) for row in w3_array]
    
    # open the file in write mode
    with open('data/w3_array.csv', 'w', newline='') as f:
        #create the csv writer
        writer = csv.writer(f)
        # write the values as a row to the csv file
        writer.writerow(w3_array)

    # convert each element of the array to float
    wo_array = [float(row) for row in wo_array]
    
    # open the file in write mode
    with open('data/wo_array.csv', 'w', newline='') as f:
        #create the csv writer
        writer = csv.writer(f)
        # write the values as a row to the csv file
        writer.writerow(wo_array)

    # convert each element of the array to float
    bias1 = [float(row) for row in bias1]
    
    # open the file in write mode
    with open('data/bias1.csv', 'w', newline='') as f:
        #create the csv writer
        writer = csv.writer(f)
        # write the values as a row to the csv file
        writer.writerow(bias1)

    # convert each element of the array to float
    bias2 = [float(row) for row in bias2]
    
    # open the file in write mode
    with open('data/bias2.csv', 'w', newline='') as f:
        #create the csv writer
        writer = csv.writer(f)
        # write the values as a row to the csv file
        writer.writerow(bias2)
    

if __name__ == '__main__':
    
    matris_train()






